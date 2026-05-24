"""Oracle TAS evaluation (Phase 2D).

Runs the steering intervention with detector = ORACLE:
  - on the verified PTC subset, treat detector score c = 1 (always steer)
  - on a matched non-conflict CONTROL subset, treat c = 0 (never steer) when
    measuring preservation; also run the always-on variant to quantify
    collateral damage on records the detector ideally would not touch.

For each value of α we report:
  - OPR_after, Recovery_after, PTC_rate_after on the PTC subset
  - PA (preservation accuracy) on the non-conflict control set
  - Mean Δ log-prob of a_new on the PTC subset (continuous benefit)
  - Mean log-prob shift on the control set (collateral damage)

The α* recommendation is the value that maximizes a simple objective:
  J(α) = Recovery_after(α) - λ · (1 - PA(α))
with λ = 1 by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

import torch

from temporal_conflict.models import LoadedModel
from temporal_conflict.schema import PTCInstance
from temporal_conflict.steering.steer import SteeringVectors, score_with_steering


def _pick_direction(
    vecs: SteeringVectors,
    variant: str,
    instance: PTCInstance,
) -> torch.Tensor:
    """Select the appropriate Δ for this instance.

    - variant="global"   -> vecs.global_delta (same for every instance)
    - variant="relation" -> vecs.per_relation[instance.relation_pid] if present,
                            else falls back to global
    - variant="domain"   -> vecs.per_domain[instance.domain] if present,
                            else falls back to global
    """
    return vecs.select(
        variant=variant,
        relation_pid=instance.relation_pid,
        domain=instance.domain,
    )


@dataclass
class OneAlphaResult:
    alpha: float
    # PTC subset
    n_ptc: int
    recovery_after: float
    opr_after: float
    ptc_rate_after: float
    mean_delta_new_ptc: float
    mean_delta_old_ptc: float
    # Control subset
    n_control: int
    preservation_accuracy: float
    mean_delta_top_control: float

    def objective(self, lam: float = 1.0) -> float:
        # Penalize lost preservation (1 - PA), reward recovery.
        return self.recovery_after - lam * (1.0 - self.preservation_accuracy)

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "ptc": {
                "n": self.n_ptc,
                "recovery_after": self.recovery_after,
                "opr_after": self.opr_after,
                "ptc_rate_after": self.ptc_rate_after,
                "mean_delta_new": self.mean_delta_new_ptc,
                "mean_delta_old": self.mean_delta_old_ptc,
            },
            "control": {
                "n": self.n_control,
                "preservation_accuracy": self.preservation_accuracy,
                "mean_delta_top": self.mean_delta_top_control,
            },
        }


@dataclass
class OracleTASResults:
    model_name: str
    layer: int
    alphas: list[float]
    by_alpha: list[OneAlphaResult] = field(default_factory=list)
    alpha_star: float = 0.0
    lambda_: float = 1.0

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "layer": self.layer,
            "alphas": self.alphas,
            "lambda": self.lambda_,
            "alpha_star": self.alpha_star,
            "by_alpha": [r.to_dict() for r in self.by_alpha],
        }


@torch.no_grad()
def _baseline_scores(
    loaded: LoadedModel,
    instance: PTCInstance,
    answer_suffix: str,
    candidate_prefix: str,
) -> tuple[float, float]:
    """Length-normalized mean log-probs of (a_old, a_new) under standard prompt
    with NO intervention. Returns (mean_lp_old, mean_lp_new).
    """
    from temporal_conflict.steering.locate import _score_candidate_with_hook, _mean_lp

    std_prefix = instance.prompt_standard + answer_suffix
    old_sum, old_n = _score_candidate_with_hook(
        loaded, std_prefix, instance.a_old_label, candidate_prefix=candidate_prefix,
    )
    new_sum, new_n = _score_candidate_with_hook(
        loaded, std_prefix, instance.a_new_label, candidate_prefix=candidate_prefix,
    )
    return _mean_lp(old_sum, old_n), _mean_lp(new_sum, new_n)


@torch.no_grad()
def _steered_scores(
    loaded: LoadedModel,
    instance: PTCInstance,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    answer_suffix: str,
    candidate_prefix: str,
) -> tuple[float, float]:
    """Length-normalized mean log-probs of (a_old, a_new) under standard prompt
    with the steering hook installed at `layer` (last prompt token).
    """
    std_prefix = instance.prompt_standard + answer_suffix
    old_sum, old_n = score_with_steering(
        loaded, std_prefix, instance.a_old_label, layer, direction, alpha,
        candidate_prefix=candidate_prefix,
    )
    new_sum, new_n = score_with_steering(
        loaded, std_prefix, instance.a_new_label, layer, direction, alpha,
        candidate_prefix=candidate_prefix,
    )
    return old_sum / old_n, new_sum / new_n


def evaluate_alpha(
    loaded: LoadedModel,
    ptc_instances: list[PTCInstance],
    control_instances: list[PTCInstance],
    layer: int,
    vecs: SteeringVectors,
    alpha: float,
    variant: str = "global",
    answer_suffix: str = "\nAnswer:",
    candidate_prefix: str = " ",
) -> OneAlphaResult:
    """Evaluate ORACLE TAS at one α: always-on on PTC subset, always-on on
    control subset (so we measure collateral damage; preservation accuracy is
    computed against the control set's unsteered argmax).

    `variant` ∈ {"global", "relation", "domain"} -- for V2/V3 the per-instance
    Δ is selected by relation/domain via _pick_direction.
    """
    n_ptc = len(ptc_instances)
    n_ctrl = len(control_instances)

    # PTC subset.
    rec_after = 0
    opr_after = 0
    ptc_after = 0
    sum_d_new = 0.0
    sum_d_old = 0.0
    for inst in ptc_instances:
        direction = _pick_direction(vecs, variant, inst)
        lp_old_base, lp_new_base = _baseline_scores(
            loaded, inst, answer_suffix, candidate_prefix,
        )
        lp_old_steer, lp_new_steer = _steered_scores(
            loaded, inst, layer, direction, alpha, answer_suffix, candidate_prefix,
        )
        if lp_new_steer > lp_old_steer:
            rec_after += 1
        if lp_old_steer > lp_new_steer:
            opr_after += 1
        # ptc_after asks: does PTC still hold? (prefers_old_std ∧ recovers_new_steered)
        # We use baseline old vs new under STANDARD prompt for the OPR-side; since
        # this is the verified PTC subset, prefers_old_under_standard is True by
        # construction (it was selected from is_ptc=True). So PTC after steering
        # is: did we flip new>old under STEERING?  (recovery via steering)
        if lp_new_steer > lp_old_steer:
            ptc_after += 1  # i.e. TAS successfully "recovered" the new fact
        sum_d_new += (lp_new_steer - lp_new_base)
        sum_d_old += (lp_old_steer - lp_old_base)

    # Control subset (preservation).
    preserved = 0
    sum_d_top = 0.0
    for inst in control_instances:
        direction = _pick_direction(vecs, variant, inst)
        lp_old_base, lp_new_base = _baseline_scores(
            loaded, inst, answer_suffix, candidate_prefix,
        )
        lp_old_steer, lp_new_steer = _steered_scores(
            loaded, inst, layer, direction, alpha, answer_suffix, candidate_prefix,
        )
        # Argmax over {old, new} under baseline and steering; preservation =
        # same argmax.
        base_top = "old" if lp_old_base > lp_new_base else "new"
        steer_top = "old" if lp_old_steer > lp_new_steer else "new"
        if base_top == steer_top:
            preserved += 1
        # Track mean change in top candidate's log-prob (a soft proxy for
        # distortion).
        base_top_lp = max(lp_old_base, lp_new_base)
        steer_top_lp = lp_old_steer if base_top == "old" else lp_new_steer
        sum_d_top += (steer_top_lp - base_top_lp)

    return OneAlphaResult(
        alpha=alpha,
        n_ptc=n_ptc,
        recovery_after=rec_after / n_ptc if n_ptc else 0.0,
        opr_after=opr_after / n_ptc if n_ptc else 0.0,
        ptc_rate_after=ptc_after / n_ptc if n_ptc else 0.0,
        mean_delta_new_ptc=sum_d_new / n_ptc if n_ptc else 0.0,
        mean_delta_old_ptc=sum_d_old / n_ptc if n_ptc else 0.0,
        n_control=n_ctrl,
        preservation_accuracy=preserved / n_ctrl if n_ctrl else 0.0,
        mean_delta_top_control=sum_d_top / n_ctrl if n_ctrl else 0.0,
    )


def sweep_alphas(
    loaded: LoadedModel,
    ptc_instances: list[PTCInstance],
    control_instances: list[PTCInstance],
    layer: int,
    vecs: SteeringVectors,
    alphas: Iterable[float],
    lam: float = 1.0,
    variant: str = "global",
    answer_suffix: str = "\nAnswer:",
    candidate_prefix: str = " ",
) -> OracleTASResults:
    """Run the α sweep and pick α* = argmax_α [Recovery(α) - λ·(1 - PA(α))]."""
    alphas = list(alphas)
    results = OracleTASResults(
        model_name=loaded.name,
        layer=layer,
        alphas=alphas,
        lambda_=lam,
    )
    for a in alphas:
        r = evaluate_alpha(
            loaded, ptc_instances, control_instances,
            layer, vecs, a, variant=variant,
            answer_suffix=answer_suffix, candidate_prefix=candidate_prefix,
        )
        results.by_alpha.append(r)
        print(f"  α={a:>5.2f}  Recovery={r.recovery_after:.3f}  "
              f"PTC_after={r.ptc_rate_after:.3f}  PA={r.preservation_accuracy:.3f}  "
              f"Δlp(new)={r.mean_delta_new_ptc:+.3f}  "
              f"Δlp(top|ctrl)={r.mean_delta_top_control:+.3f}", flush=True)

    if results.by_alpha:
        best = max(results.by_alpha, key=lambda r: r.objective(lam))
        results.alpha_star = best.alpha

    return results
