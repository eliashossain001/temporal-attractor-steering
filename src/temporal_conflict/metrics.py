"""Aggregate metrics for Phase 1 PTC screening.

We report three core rates and one elicitation-gap measure:

  OPR           outdated preference rate under standard prompting
                (fraction with prefers_old_under_standard)
  recovery_rate fraction where the temporal cue flips preference to a_new
  ptc_rate      fraction satisfying the full PTC definition (both above)
  EG (logprob)  mean over instances of mean_lp(a_new|temporal) - mean_lp(a_new|standard)
  EG (prob)     same gap expressed in per-token probability space
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp
from typing import Callable, Iterable

from temporal_conflict.schema import InstanceResult


@dataclass
class AggregateMetrics:
    n: int
    opr: float
    recovery_rate: float
    ptc_rate: float
    elicitation_gap_logprob: float
    elicitation_gap_prob: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "opr": self.opr,
            "recovery_rate": self.recovery_rate,
            "ptc_rate": self.ptc_rate,
            "elicitation_gap_logprob": self.elicitation_gap_logprob,
            "elicitation_gap_prob": self.elicitation_gap_prob,
        }


def aggregate(results: Iterable[InstanceResult]) -> AggregateMetrics:
    results = list(results)
    n = len(results)
    if n == 0:
        return AggregateMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    opr = sum(r.prefers_old_under_standard for r in results) / n
    recovery = sum(r.recovers_new_under_temporal for r in results) / n
    ptc = sum(r.is_ptc for r in results) / n

    gap_lp = sum(r.tmp_new.mean_logprob - r.std_new.mean_logprob for r in results) / n
    gap_p = sum(
        exp(r.tmp_new.mean_logprob) - exp(r.std_new.mean_logprob) for r in results
    ) / n

    return AggregateMetrics(
        n=n,
        opr=opr,
        recovery_rate=recovery,
        ptc_rate=ptc,
        elicitation_gap_logprob=gap_lp,
        elicitation_gap_prob=gap_p,
    )


def aggregate_grouped(
    results: Iterable[InstanceResult],
    key_fn: Callable[[InstanceResult], str],
) -> dict[str, AggregateMetrics]:
    """Aggregate metrics by a grouping key (e.g. relation_pid, domain)."""
    groups: dict[str, list[InstanceResult]] = defaultdict(list)
    for r in results:
        groups[key_fn(r)].append(r)
    return {k: aggregate(v) for k, v in groups.items()}
