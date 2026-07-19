"""WP-B: steering-control ablations (checklist item 2).

Two negative controls defend the Stage-2 localisation claim by holding the
intervention magnitude fixed and changing only *what* / *where* we steer:

  * ``random_direction`` -- at the conflict-critical layer ell*, replace the
    verified-conflict direction with a random unit vector rescaled to the same
    L2 norm as the Delta it displaces, at the same alpha. If Recovery collapses
    to chance, the recovery is carried by the *direction*, not by generic
    activation perturbation of that norm.

  * ``wrong_layer`` -- apply the genuine ell* Delta at an early "null" layer
    (same vector, same alpha). If Recovery collapses, the effect is specific to
    ell* rather than reproducible anywhere in the residual stream.

Both are evaluated always-on (oracle gate) at the model's alpha*, on the same
PTC and control subsets as the oracle TAS sweep, so the numbers drop straight
into the localisation ablation table next to oracle V2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from temporal_conflict.models import LoadedModel
from temporal_conflict.schema import PTCInstance
from temporal_conflict.steering.eval import _baseline_scores, _pick_direction, _steered_scores
from temporal_conflict.steering.steer import SteeringVectors


@dataclass
class ControlConfig:
    alpha: float
    ell_star: int
    null_layer: int
    variant: str = "relation"
    answer_suffix: str = "\nAnswer:"
    candidate_prefix: str = " "
    seed: int = 0


def _random_like(direction: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """A random vector with the same L2 norm as ``direction``."""
    d = direction.shape[0]
    v = torch.from_numpy(rng.standard_normal(d)).to(direction.dtype)
    v = v / (v.norm() + 1e-8) * direction.norm()
    return v


def _recovery_and_pa(
    loaded: LoadedModel,
    ptc_instances: list[PTCInstance],
    control_instances: list[PTCInstance],
    layer: int,
    direction_fn,
    cfg: ControlConfig,
) -> tuple[list[dict], list[dict]]:
    """Score a control on PTC (recovery) and control (preservation) subsets.

    ``direction_fn(inst, base_direction) -> direction`` lets each control swap
    the vector while keeping norm matching in one place.
    """
    ptc_rows: list[dict] = []
    for inst in ptc_instances:
        base = _pick_direction_safe(cfg.variant, inst)
        direction = direction_fn(inst, base)
        old_lp, new_lp = _steered_scores(
            loaded, inst, layer, direction, cfg.alpha,
            cfg.answer_suffix, cfg.candidate_prefix,
        )
        ptc_rows.append({
            "instance_id": inst.instance_id,
            "relation_pid": inst.relation_pid,
            "domain": inst.domain,
            "subset": "ptc",
            "old_lp": old_lp,
            "new_lp": new_lp,
            "recovered": int(new_lp > old_lp),
        })

    ctrl_rows: list[dict] = []
    for inst in control_instances:
        base = _pick_direction_safe(cfg.variant, inst)
        direction = direction_fn(inst, base)
        old_base, new_base = _baseline_scores(
            loaded, inst, cfg.answer_suffix, cfg.candidate_prefix,
        )
        old_lp, new_lp = _steered_scores(
            loaded, inst, layer, direction, cfg.alpha,
            cfg.answer_suffix, cfg.candidate_prefix,
        )
        ctrl_rows.append({
            "instance_id": inst.instance_id,
            "relation_pid": inst.relation_pid,
            "domain": inst.domain,
            "subset": "control",
            "old_lp": old_lp,
            "new_lp": new_lp,
            "preserved": int((new_lp > old_lp) == (new_base > old_base)),
        })
    return ptc_rows, ctrl_rows


# _pick_direction needs the vecs object; bind it lazily via a closure set in
# evaluate_controls so the two-control loop above stays readable.
_VECS: SteeringVectors | None = None


def _pick_direction_safe(variant: str, inst: PTCInstance) -> torch.Tensor:
    assert _VECS is not None, "steering vectors not bound"
    return _pick_direction(_VECS, variant, inst)


def evaluate_controls(
    loaded: LoadedModel,
    vecs: SteeringVectors,
    ptc_instances: list[PTCInstance],
    control_instances: list[PTCInstance],
    cfg: ControlConfig,
) -> dict:
    """Run the random-direction and wrong-layer controls and summarise."""
    global _VECS
    _VECS = vecs
    rng = np.random.default_rng(cfg.seed)

    def summarise(ptc_rows: list[dict], ctrl_rows: list[dict]) -> dict:
        rec = (sum(r["recovered"] for r in ptc_rows) / len(ptc_rows)) if ptc_rows else 0.0
        pa = (sum(r["preserved"] for r in ctrl_rows) / len(ctrl_rows)) if ctrl_rows else 0.0
        return {"n_ptc": len(ptc_rows), "recovery": rec,
                "n_control": len(ctrl_rows), "preservation_accuracy": pa}

    results: dict[str, dict] = {}
    per_instance: dict[str, dict] = {}

    # Control 1: random direction at ell*, norm-matched to the displaced Delta.
    def random_dir(inst, base):
        return _random_like(base, rng)

    ptc_r, ctrl_r = _recovery_and_pa(
        loaded, ptc_instances, control_instances, cfg.ell_star, random_dir, cfg,
    )
    results["random_direction"] = summarise(ptc_r, ctrl_r)
    per_instance["random_direction"] = {"ptc": ptc_r, "control": ctrl_r}

    # Control 2: genuine Delta applied at the null layer.
    def real_dir(inst, base):
        return base

    ptc_w, ctrl_w = _recovery_and_pa(
        loaded, ptc_instances, control_instances, cfg.null_layer, real_dir, cfg,
    )
    results["wrong_layer"] = summarise(ptc_w, ctrl_w)
    per_instance["wrong_layer"] = {"ptc": ptc_w, "control": ctrl_w}

    _VECS = None
    return {
        "model": loaded.name,
        "config": {
            "alpha": cfg.alpha, "ell_star": cfg.ell_star,
            "null_layer": cfg.null_layer, "variant": cfg.variant, "seed": cfg.seed,
        },
        "summary": results,
        "per_instance": per_instance,
    }


def write_results(path: str | Path, results: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2))
