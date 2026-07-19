"""Driver for the prompt-baseline (WP-A) and steering-control (WP-B) evaluations.

Reconstructs the same verified-PTC subset and non-conflict control set used by
the Phase 2F TAS evaluation directly from the screening JSONL, so the baseline
numbers are computed on identical records. WP-B additionally needs the ell*
steering vectors; if they are not supplied it is skipped and only the prompt
baselines are produced.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from temporal_conflict.config import ModelSpec, PromptingSpec
from temporal_conflict.env import get_hf_token
from temporal_conflict.io import load_directory
from temporal_conflict.models import load_model
from temporal_conflict.schema import PTCInstance
from temporal_conflict.baselines.prompts import (
    NEW_PROMPTS,
    PromptBaselineConfig,
    evaluate_prompt_baselines,
    write_results as write_prompt_results,
)
from temporal_conflict.baselines.controls import (
    ControlConfig,
    evaluate_controls,
    write_results as write_control_results,
)


def _load_screening(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with Path(path).open() as f:
        for line in f:
            r = json.loads(line)
            out[r["instance_id"]] = r
    return out


def _subsets(
    screening: dict[str, dict],
    instances: list[PTCInstance],
    recovery_threshold: float,
    control_size: int,
    seed: int = 0,
) -> tuple[list[PTCInstance], list[PTCInstance]]:
    """Verified-PTC subset and matched non-conflict control subset."""
    ptc_ids: set[str] = set()
    non_ptc_ids: list[str] = []
    for iid, r in screening.items():
        if r.get("is_ptc"):
            lp = r["scores"]["temporal"]["a_new"]["mean_logprob"]
            if lp >= recovery_threshold:
                ptc_ids.add(iid)
        else:
            non_ptc_ids.append(iid)

    rng = random.Random(seed)
    rng.shuffle(non_ptc_ids)
    control_ids = set(non_ptc_ids[:control_size])

    by_id = {i.instance_id: i for i in instances}
    ptc = [by_id[i] for i in ptc_ids if i in by_id]
    control = [by_id[i] for i in control_ids if i in by_id]
    return ptc, control


def run_baselines(
    *,
    model_spec: ModelSpec,
    prompting: PromptingSpec,
    screening_path: Path,
    data_dir: Path,
    out_dir: Path,
    device_map: str = "auto",
    control_size: int = 500,
    recovery_threshold: float = -3.0,
    limit: int | None = None,
    run_prompts: bool = True,
    steering_vectors_path: Path | None = None,
    afr_path: Path | None = None,
    alpha: float | None = None,
    null_layer: int | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    screening = _load_screening(Path(screening_path))
    instances = load_directory(Path(data_dir))
    ptc, control = _subsets(
        screening, instances, recovery_threshold, control_size,
    )
    if limit is not None:
        ptc = ptc[:limit]
        control = control[:limit]
    print(f"[baselines] {model_spec.key}: PTC={len(ptc)} control={len(control)}",
          flush=True)

    loaded = load_model(
        name=model_spec.key, hf_id=model_spec.hf_id, dtype=model_spec.dtype,
        device_map=device_map, trust_remote_code=model_spec.trust_remote_code,
        token=get_hf_token(),
    )

    if run_prompts:
        cfg = PromptBaselineConfig(
            prompts=NEW_PROMPTS,
            answer_suffix=prompting.answer_suffix,
            candidate_prefix=prompting.candidate_prefix,
        )
        res = evaluate_prompt_baselines(loaded, ptc, control, screening, cfg)
        write_prompt_results(out_dir / "prompt_baselines.json", res)
        for name, s in res["summary"].items():
            print(f"  [prompt:{name}] Recovery={s['recovery']:.3f} "
                  f"PA={s['preservation_accuracy']:.3f}", flush=True)

    if steering_vectors_path is not None and afr_path is not None:
        from temporal_conflict.steering.steer import load_vectors
        vecs = load_vectors(str(steering_vectors_path))
        afr = json.loads(Path(afr_path).read_text())
        ell_star = int(afr["ell_star"])
        a = alpha if alpha is not None else _alpha_star(out_dir, afr_path)
        nl = null_layer if null_layer is not None else max(2, ell_star // 8)
        ccfg = ControlConfig(
            alpha=a, ell_star=ell_star, null_layer=nl,
            variant="relation",
            answer_suffix=prompting.answer_suffix,
            candidate_prefix=prompting.candidate_prefix,
        )
        cres = evaluate_controls(loaded, vecs, ptc, control, ccfg)
        write_control_results(out_dir / "steering_controls.json", cres)
        for name, s in cres["summary"].items():
            print(f"  [control:{name}] Recovery={s['recovery']:.3f} "
                  f"PA={s['preservation_accuracy']:.3f}", flush=True)
    else:
        print("  [control] skipped (no steering vectors / afr profile supplied)",
              flush=True)


def _alpha_star(out_dir: Path, afr_path: Path) -> float:
    """Best-effort alpha*: look for oracle_tas_relation.json beside the afr."""
    cand = Path(afr_path).parent / "oracle_tas_relation.json"
    if cand.exists():
        return float(json.loads(cand.read_text())["alpha_star"])
    raise ValueError("alpha not supplied and oracle_tas_relation.json not found")
