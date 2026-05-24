"""Pipeline stage classes.

Each stage is a small class with `requires` (inputs that must exist on
disk), `produces` (outputs it will write), and a `run` method. Stages are
idempotent: if every output already exists they short-circuit and return
the cached artifacts.

The classes are thin wrappers around the lower-level functional modules
in `temporal_conflict.*`; we keep the science code there and use these
classes as orchestration / serialization boilerplate.
"""

from __future__ import annotations

import json
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from temporal_conflict.config import ModelSpec, PromptingSpec, RunPaths
from temporal_conflict.env import get_hf_token
from temporal_conflict.io import load_directory, write_jsonl
from temporal_conflict.metrics import aggregate, aggregate_grouped
from temporal_conflict.models import LoadedModel, load_model
from temporal_conflict.schema import InstanceResult, PTCInstance
from temporal_conflict.steering.activations import (
    extract_many,
    load_activations,
    save_activations,
)
from temporal_conflict.steering.detector import (
    save_detector,
    train_detector,
)
from temporal_conflict.steering.eval import sweep_alphas
from temporal_conflict.steering.locate import (
    critical_layer,
    layer_sweep,
    summarize_sweep,
)
from temporal_conflict.steering.steer import (
    SteeringVectors,
    build_vectors,
    load_vectors,
    save_vectors,
)
from temporal_conflict.steering.detector import load_detector
from temporal_conflict.steering.tas import TASConfig, tas_score_candidates
from temporal_conflict.verify import verify_instance


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


@dataclass
class StageContext:
    """Per-run state shared by every stage of a single model's pipeline."""

    model_spec: ModelSpec
    prompting: PromptingSpec
    paths: RunPaths
    data_dir: Path
    device_map: str = "auto"
    hf_token: str | None = None

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.hf_token is None:
            self.hf_token = get_hf_token()


class Stage:
    """Base class. Subclasses must define `name` and override `_run`."""

    name: str = "stage"

    def __init__(self, ctx: StageContext) -> None:
        self.ctx = ctx

    # Subclasses override.
    def outputs(self) -> list[Path]:
        return []

    def is_cached(self) -> bool:
        outs = self.outputs()
        return bool(outs) and all(p.exists() for p in outs)

    def run(self, *, force: bool = False) -> None:
        if not force and self.is_cached():
            self._log(f"[skip] {self.name}: outputs already exist")
            return
        self._log(f"=== {self.name} ===")
        t0 = time.time()
        self._run()
        self._log(f"=== {self.name} done in {time.time() - t0:.1f}s ===")

    def _run(self) -> None:
        raise NotImplementedError

    def _log(self, msg: str) -> None:
        print(msg, flush=True)

    # Convenience: lazily load the model only when a stage needs it.
    def load_model(self) -> LoadedModel:
        spec = self.ctx.model_spec
        self._log(f"Loading {spec.key} ({spec.hf_id})...")
        t0 = time.time()
        m = load_model(
            name=spec.key,
            hf_id=spec.hf_id,
            dtype=spec.dtype,
            device_map=self.ctx.device_map,
            trust_remote_code=spec.trust_remote_code,
            token=self.ctx.hf_token,
        )
        self._log(f"Loaded in {time.time() - t0:.1f}s; device={m.device}")
        return m


# ---------------------------------------------------------------------------
# Phase 1: screening
# ---------------------------------------------------------------------------


class ScreeningStage(Stage):
    """Phase 1: score the four log-probs per instance and apply PTC."""

    name = "Phase 1: PTC screening"

    def outputs(self) -> list[Path]:
        return [self.ctx.paths.screening_jsonl, self.ctx.paths.screening_summary]

    def _run(self) -> None:
        instances = load_directory(self.ctx.data_dir)
        if not instances:
            raise RuntimeError(f"No PTC instances found under {self.ctx.data_dir}")
        self._log(f"Loaded {len(instances)} instances from {self.ctx.data_dir}")

        model = self.load_model()
        results: list[InstanceResult] = []
        prompting = self.ctx.prompting
        for i, inst in enumerate(instances, start=1):
            results.append(
                verify_instance(
                    model, inst,
                    answer_suffix=prompting.answer_suffix,
                    candidate_prefix=prompting.candidate_prefix,
                )
            )
            if i % 100 == 0:
                self._log(f"  scored {i}/{len(instances)}")

        write_jsonl(self.ctx.paths.screening_jsonl, (r.to_dict() for r in results))

        overall = aggregate(results)
        by_relation = aggregate_grouped(results, key_fn=lambda r: r.relation_pid)
        by_domain = aggregate_grouped(results, key_fn=lambda r: r.domain)
        summary = {
            "model": self.ctx.model_spec.key,
            "hf_id": self.ctx.model_spec.hf_id,
            "n_instances": len(results),
            "overall": overall.to_dict(),
            "by_relation": {k: v.to_dict() for k, v in by_relation.items()},
            "by_domain": {k: v.to_dict() for k, v in by_domain.items()},
        }
        self.ctx.paths.screening_summary.write_text(json.dumps(summary, indent=2))
        self._log(
            f"PTC rate: {overall.ptc_rate:.3f}  "
            f"OPR: {overall.opr:.3f}  "
            f"Recovery: {overall.recovery_rate:.3f}"
        )


# ---------------------------------------------------------------------------
# Phase 2A: activation cache
# ---------------------------------------------------------------------------


class ActivationCacheStage(Stage):
    """Phase 2A: cache per-layer residual-stream activations on the verified
    PTC subset.
    """

    name = "Phase 2A: activation cache"

    def __init__(
        self,
        ctx: StageContext,
        recovery_threshold: float = -3.0,
        sample_cap: int | None = 500,
        seed: int = 42,
    ) -> None:
        super().__init__(ctx)
        self.recovery_threshold = recovery_threshold
        self.sample_cap = sample_cap
        self.seed = seed

    def outputs(self) -> list[Path]:
        return [self.ctx.paths.activations]

    def _filtered_ptc_ids(self) -> set[str]:
        ids: set[str] = set()
        with self.ctx.paths.screening_jsonl.open() as f:
            for line in f:
                r = json.loads(line)
                if not r.get("is_ptc"):
                    continue
                lp = r["scores"]["temporal"]["a_new"]["mean_logprob"]
                if lp < self.recovery_threshold:
                    continue
                ids.add(r["instance_id"])
        return ids

    def _run(self) -> None:
        keep_ids = self._filtered_ptc_ids()
        self._log(f"Verified PTC subset size: {len(keep_ids)}")
        if not keep_ids:
            raise RuntimeError("Empty filtered PTC subset; nothing to cache.")

        all_instances = load_directory(self.ctx.data_dir)
        instances = [i for i in all_instances if i.instance_id in keep_ids]
        if self.sample_cap is not None and len(instances) > self.sample_cap:
            rng = random.Random(self.seed)
            instances = sorted(instances, key=lambda i: i.instance_id)
            instances = rng.sample(instances, self.sample_cap)
            self._log(f"Sampled {len(instances)} of {len(keep_ids)} (seed={self.seed})")
        self._log(f"Extracting activations for {len(instances)} instances")

        model = self.load_model()
        records = extract_many(
            model, instances,
            answer_suffix=self.ctx.prompting.answer_suffix,
        )
        save_activations(str(self.ctx.paths.activations), records)
        mb = self.ctx.paths.activations.stat().st_size / (1024 * 1024)
        self._log(f"Wrote {self.ctx.paths.activations} ({mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Phase 2B: layer locator
# ---------------------------------------------------------------------------


class LayerLocatorStage(Stage):
    """Phase 2B: activation-patching sweep, write AFR profile + pick ell*."""

    name = "Phase 2B: layer locator"

    def __init__(self, ctx: StageContext, layers: list[int] | None = None) -> None:
        super().__init__(ctx)
        self.layers = layers  # None -> all layers

    def outputs(self) -> list[Path]:
        return [self.ctx.paths.afr_profile]

    def _run(self) -> None:
        cache = load_activations(str(self.ctx.paths.activations))
        ids: list[str] = cache["instance_ids"]
        temporal: torch.Tensor = cache["temporal"]  # [N, L, d]
        all_instances = load_directory(self.ctx.data_dir)
        by_id = {i.instance_id: i for i in all_instances}
        instances = [by_id[i] for i in ids if i in by_id]
        if len(instances) != len(ids):
            self._log(
                f"[warn] {len(ids) - len(instances)} cached ids missing from data; "
                "proceeding with intersection."
            )
            temporal = temporal[: len(instances)]

        L = temporal.shape[1]
        layers = self.layers if self.layers is not None else list(range(L))

        model = self.load_model()
        per_layer = layer_sweep(
            model, instances, temporal, layers=layers,
            answer_suffix=self.ctx.prompting.answer_suffix,
            candidate_prefix=self.ctx.prompting.candidate_prefix,
        )
        summary = summarize_sweep(per_layer)
        ell_star = critical_layer(summary)

        payload = {
            "model": self.ctx.model_spec.key,
            "n_instances": len(instances),
            "layers": layers,
            "per_layer": {str(k): v for k, v in summary.items()},
            "ell_star": int(ell_star),
        }
        self.ctx.paths.afr_profile.write_text(json.dumps(payload, indent=2))
        self._log(
            f"ell* = {ell_star}  AFR = {summary[ell_star]['afr']:.3f}"
        )


# ---------------------------------------------------------------------------
# Phase 2C: steering vector construction (V1/V2/V3)
# ---------------------------------------------------------------------------


class SteeringVectorBuilderStage(Stage):
    """Phase 2C: build V1/V2/V3 steering vectors at ell*."""

    name = "Phase 2C: steering vector construction"

    def outputs(self) -> list[Path]:
        return [self.ctx.paths.steering_vectors]

    def _run(self) -> None:
        afr = json.loads(self.ctx.paths.afr_profile.read_text())
        ell_star = int(afr["ell_star"])
        cache = load_activations(str(self.ctx.paths.activations))
        vectors = build_vectors(
            layer=ell_star,
            standard_acts=cache["standard"],
            temporal_acts=cache["temporal"],
            relation_pids=cache["relation_pids"],
            domains=cache["domains"],
            compute_per_relation=True,
            compute_per_domain=True,
        )
        save_vectors(str(self.ctx.paths.steering_vectors), vectors)
        self._log(
            f"Wrote V1/V2/V3 at layer {ell_star}; "
            f"per_relation={len(vectors.per_relation)}, "
            f"per_domain={len(vectors.per_domain)}"
        )


# ---------------------------------------------------------------------------
# Phase 2D: oracle TAS alpha sweep
# ---------------------------------------------------------------------------


class OracleTASStage(Stage):
    """Phase 2D: oracle (always-on) alpha sweep on PTC + control sets."""

    name = "Phase 2D: oracle TAS alpha sweep"

    def __init__(
        self,
        ctx: StageContext,
        alphas: list[float] | None = None,
        variant: str = "relation",
        control_size: int = 200,
    ) -> None:
        super().__init__(ctx)
        self.alphas = alphas or [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
        self.variant = variant
        self.control_size = control_size

    def outputs(self) -> list[Path]:
        return [self.ctx.paths.oracle_tas]

    def _control_set(
        self, all_instances: list[PTCInstance], ptc_ids: set[str]
    ) -> list[PTCInstance]:
        """Pick a matched non-conflict control subset."""
        screening = {}
        with self.ctx.paths.screening_jsonl.open() as f:
            for line in f:
                r = json.loads(line)
                screening[r["instance_id"]] = r
        non_ptc_ids = [
            iid for iid, r in screening.items()
            if iid not in ptc_ids and not r.get("is_ptc", False)
        ]
        rng = random.Random(0)
        rng.shuffle(non_ptc_ids)
        pick = set(non_ptc_ids[: self.control_size])
        return [i for i in all_instances if i.instance_id in pick]

    def _run(self) -> None:
        vectors = load_vectors(str(self.ctx.paths.steering_vectors))
        cache = load_activations(str(self.ctx.paths.activations))
        ptc_ids: set[str] = set(cache["instance_ids"])
        all_instances = load_directory(self.ctx.data_dir)
        ptc_instances = [i for i in all_instances if i.instance_id in ptc_ids]
        control_instances = self._control_set(all_instances, ptc_ids)
        self._log(
            f"PTC = {len(ptc_instances)}  control = {len(control_instances)}"
        )

        model = self.load_model()
        results = sweep_alphas(
            model, ptc_instances, control_instances,
            layer=vectors.layer, vecs=vectors,
            alphas=self.alphas, variant=self.variant,
            answer_suffix=self.ctx.prompting.answer_suffix,
            candidate_prefix=self.ctx.prompting.candidate_prefix,
        )
        self.ctx.paths.oracle_tas.write_text(json.dumps(results.to_dict(), indent=2))
        self._log(f"alpha* = {results.alpha_star}")


# ---------------------------------------------------------------------------
# Phase 2E: detector training
# ---------------------------------------------------------------------------


class DetectorStage(Stage):
    """Phase 2E: extract detector activations (PTC + matched negatives) and
    train a calibrated logistic probe at ell*.
    """

    name = "Phase 2E: detector training"

    def __init__(self, ctx: StageContext, max_negatives: int = 1000) -> None:
        super().__init__(ctx)
        self.max_negatives = max_negatives

    def outputs(self) -> list[Path]:
        return [
            self.ctx.paths.detector_activations,
            self.ctx.paths.detector,
            self.ctx.paths.detector_metrics,
        ]

    def _collect_activations(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Two passes through the screening JSONL: one to pick PTC + negative
        ids, one to forward through the model and capture h_layer.
        """
        from temporal_conflict.steering.hooks import decoder_layer, make_capture_hook

        screening: dict[str, dict] = {}
        with self.ctx.paths.screening_jsonl.open() as f:
            for line in f:
                r = json.loads(line)
                screening[r["instance_id"]] = r
        ptc_ids = [iid for iid, r in screening.items() if r.get("is_ptc")]
        neg_ids = [iid for iid, r in screening.items() if not r.get("is_ptc")]
        rng = random.Random(0)
        rng.shuffle(neg_ids)
        neg_ids = neg_ids[: self.max_negatives]
        self._log(f"PTC pos = {len(ptc_ids)}  matched neg = {len(neg_ids)}")

        keep = set(ptc_ids) | set(neg_ids)
        all_instances = load_directory(self.ctx.data_dir)
        by_id = {i.instance_id: i for i in all_instances if i.instance_id in keep}

        model = self.load_model()
        layer_mod = decoder_layer(model.model, layer)
        pos_acts: list[np.ndarray] = []
        neg_acts: list[np.ndarray] = []

        def capture(iid_list: list[str], target: list[np.ndarray]) -> None:
            for iid in iid_list:
                inst = by_id.get(iid)
                if inst is None:
                    continue
                text = inst.prompt_standard + self.ctx.prompting.answer_suffix
                ids = model.tokenizer(
                    text, return_tensors="pt", add_special_tokens=True
                ).input_ids.to(model.device)
                last = ids.shape[1] - 1
                buf: list[torch.Tensor] = []
                h = layer_mod.register_forward_hook(make_capture_hook(buf, position=last))
                try:
                    with torch.no_grad():
                        model.model(ids)
                finally:
                    h.remove()
                target.append(buf[0].squeeze(0).cpu().float().numpy())

        self._log("Extracting positive activations...")
        capture(ptc_ids, pos_acts)
        self._log("Extracting negative activations...")
        capture(neg_ids, neg_acts)

        pos = np.stack(pos_acts, axis=0) if pos_acts else np.zeros((0, 1))
        neg = np.stack(neg_acts, axis=0) if neg_acts else np.zeros((0, 1))
        payload = {"positives": pos, "negatives": neg, "layer": int(layer)}
        with self.ctx.paths.detector_activations.open("wb") as f:
            pickle.dump(payload, f)
        return pos, neg

    def _run(self) -> None:
        afr = json.loads(self.ctx.paths.afr_profile.read_text())
        layer = int(afr["ell_star"])
        if self.ctx.paths.detector_activations.exists():
            with self.ctx.paths.detector_activations.open("rb") as f:
                p = pickle.load(f)
            pos, neg = p["positives"], p["negatives"]
        else:
            pos, neg = self._collect_activations(layer)

        detector, metrics = train_detector(
            positive_acts=pos, negative_acts=neg, layer=layer,
        )
        save_detector(str(self.ctx.paths.detector), detector)
        self.ctx.paths.detector_metrics.write_text(json.dumps(metrics.to_dict(), indent=2))
        self._log(
            f"AUPRC = {metrics.auprc:.3f}  AUROC = {metrics.auroc:.3f}  "
            f"ECE = {metrics.ece:.3f}"
        )


# ---------------------------------------------------------------------------
# Phase 2F: full TAS evaluation
# ---------------------------------------------------------------------------


class TASEvaluationStage(Stage):
    """Phase 2F: end-to-end TAS evaluation at a given (alpha, tau)."""

    name = "Phase 2F: end-to-end TAS evaluation"

    def __init__(
        self,
        ctx: StageContext,
        tau: float,
        alpha: float | None = None,
        variant: str = "relation",
        control_size: int = 500,
    ) -> None:
        super().__init__(ctx)
        self.tau = tau
        self.alpha = alpha
        self.variant = variant
        self.control_size = control_size

    def outputs(self) -> list[Path]:
        return [self.ctx.paths.tas_eval(self.tau)]

    def _resolve_alpha(self) -> float:
        if self.alpha is not None:
            return self.alpha
        oracle = json.loads(self.ctx.paths.oracle_tas.read_text())
        return float(oracle["alpha_star"])

    def _run(self) -> None:
        alpha = self._resolve_alpha()

        detector = load_detector(str(self.ctx.paths.detector))
        vecs = load_vectors(str(self.ctx.paths.steering_vectors))
        tas_cfg = TASConfig(
            layer=vecs.layer, alpha=alpha, tau=self.tau, variant=self.variant,
        )

        screening_by_id: dict[str, dict] = {}
        with self.ctx.paths.screening_jsonl.open() as f:
            for line in f:
                r = json.loads(line)
                screening_by_id[r["instance_id"]] = r
        all_instances = load_directory(self.ctx.data_dir)
        instances = [
            i for i in all_instances if i.instance_id in screening_by_id
        ]

        model = self.load_model()
        per_instance: list[dict] = []
        for i, inst in enumerate(instances, start=1):
            prefix = inst.prompt_standard + self.ctx.prompting.answer_suffix
            direction = vecs.select(
                variant=self.variant,
                relation_pid=inst.relation_pid,
                domain=inst.domain,
            )
            scores, route = tas_score_candidates(
                model, prefix=prefix,
                candidates=[inst.a_old_label, inst.a_new_label],
                detector=detector, vecs=vecs, cfg=tas_cfg,
                direction=direction,
                candidate_prefix=self.ctx.prompting.candidate_prefix,
            )
            old_lp = scores[0][0] / max(scores[0][1], 1)
            new_lp = scores[1][0] / max(scores[1][1], 1)
            r = screening_by_id[inst.instance_id]
            per_instance.append({
                "instance_id": inst.instance_id,
                "relation_pid": inst.relation_pid,
                "domain": inst.domain,
                "is_ptc_label": bool(r.get("is_ptc", False)),
                "knowledge_absent": (
                    r["scores"]["temporal"]["a_new"]["mean_logprob"] < -3.0
                ),
                "detector_score": route.detector_score,
                "steered": route.steered,
                "effective_alpha": route.effective_alpha,
                "old_lp_tas": old_lp,
                "new_lp_tas": new_lp,
                "old_lp_std_baseline":
                    r["scores"]["standard"]["a_old"]["mean_logprob"],
                "new_lp_std_baseline":
                    r["scores"]["standard"]["a_new"]["mean_logprob"],
            })
            if i % 100 == 0:
                self._log(f"  scored {i}/{len(instances)}")

        result = _aggregate_tas(per_instance, tas_cfg, self.ctx.model_spec.key)
        out = self.ctx.paths.tas_eval(self.tau)
        out.write_text(json.dumps(result, indent=2))
        ptc_summary = result["summary"]["ptc_subset"]
        self._log(
            f"tau={self.tau}  alpha={alpha}  "
            f"recovery_tas={ptc_summary['recovery_tas']:.3f}"
        )


def _aggregate_tas(
    per_instance: list[dict], tas_cfg: TASConfig, model_key: str,
) -> dict:
    """Aggregate per-record TAS outputs into the same JSON shape used by the
    legacy script (kept stable so downstream analysis still works).
    """
    ptc = [r for r in per_instance if r["is_ptc_label"]]
    non_ptc = [r for r in per_instance if not r["is_ptc_label"]]

    def rate(rs: list[dict], key_old: str, key_new: str) -> float:
        return sum(r[key_new] > r[key_old] for r in rs) / len(rs) if rs else 0.0

    summary = {
        "model": model_key,
        "tas_config": {
            "layer": tas_cfg.layer, "alpha": tas_cfg.alpha,
            "tau": tas_cfg.tau, "variant": tas_cfg.variant,
            "scale_by_score": tas_cfg.scale_by_score,
        },
        "detector": {
            "steered_total": sum(r["steered"] for r in per_instance),
            "steered_on_ptc": sum(r["steered"] for r in ptc),
            "steered_on_non_ptc": sum(r["steered"] for r in non_ptc),
            "n_ptc": len(ptc),
            "n_non_ptc": len(non_ptc),
        },
        "ptc_subset": {
            "n": len(ptc),
            "recovery_baseline_standard":
                rate(ptc, "old_lp_std_baseline", "new_lp_std_baseline"),
            "recovery_tas": rate(ptc, "old_lp_tas", "new_lp_tas"),
        },
        "non_ptc_subset": {
            "n": len(non_ptc),
            "preservation_vs_standard_baseline": (
                sum(
                    (r["new_lp_tas"] > r["old_lp_tas"])
                    == (r["new_lp_std_baseline"] > r["old_lp_std_baseline"])
                    for r in non_ptc
                ) / len(non_ptc) if non_ptc else 0.0
            ),
        },
    }
    return {"summary": summary, "per_instance": per_instance}
