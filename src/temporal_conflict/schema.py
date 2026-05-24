"""Dataclasses for PTC instances and per-instance results."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional


@dataclass(slots=True)
class PTCInstance:
    """A single temporal-fact quadruple Q = (q, a_old, a_new, t_update).

    Mirrors the JSONL records produced by the data pipeline. Unknown fields
    are stashed in `extra` so we can round-trip without losing information.
    """

    subject_qid: str
    subject_label: str
    relation_pid: str
    relation_label: str
    domain: str
    a_old_qid: str
    a_old_label: str
    a_new_qid: str
    a_new_label: str
    t_update: str
    prompt_standard: str
    prompt_temporal: str
    # Optional temporal metadata
    t_old_start: Optional[str] = None
    t_old_end: Optional[str] = None
    t_new_start: Optional[str] = None
    t_new_end: Optional[str] = None
    coverage_tier: Optional[str] = None
    passed_filter: Optional[bool] = None
    # Catch-all for anything else in the JSONL (frequency stats, news urls, etc.)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def instance_id(self) -> str:
        return (
            f"{self.relation_pid}:{self.subject_qid}:"
            f"{self.a_old_qid}->{self.a_new_qid}"
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PTCInstance":
        known = {f.name for f in fields(cls)}
        kwargs = {k: d[k] for k in known if k in d and k != "extra"}
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs, extra=extras)


@dataclass(slots=True)
class ScoreRecord:
    """Raw log-prob measurement for one (prefix, candidate) pair."""

    sum_logprob: float
    n_tokens: int

    @property
    def mean_logprob(self) -> float:
        return self.sum_logprob / self.n_tokens if self.n_tokens > 0 else float("-inf")


@dataclass(slots=True)
class InstanceResult:
    """Four log-prob measurements for one (model, instance) plus PTC verdict.

    The PTC verdict applies Definition 2 from the proposal:
      prefers_old_under_standard:  P(a_old | q) > P(a_new | q)
      recovers_new_under_temporal: P(a_new | q, c_temp) > P(a_old | q, c_temp)
      is_ptc:                      both of the above hold.
    Probabilities are compared in length-normalized form (mean log-prob per
    candidate token) to avoid biasing toward shorter answers.
    """

    instance_id: str
    model_name: str
    relation_pid: str
    domain: str
    std_old: ScoreRecord
    std_new: ScoreRecord
    tmp_old: ScoreRecord
    tmp_new: ScoreRecord
    prefers_old_under_standard: bool
    recovers_new_under_temporal: bool
    is_ptc: bool

    def to_dict(self) -> dict[str, Any]:
        def s(r: ScoreRecord) -> dict[str, float]:
            return {
                "sum_logprob": r.sum_logprob,
                "n_tokens": r.n_tokens,
                "mean_logprob": r.mean_logprob,
            }

        return {
            "instance_id": self.instance_id,
            "model_name": self.model_name,
            "relation_pid": self.relation_pid,
            "domain": self.domain,
            "scores": {
                "standard": {"a_old": s(self.std_old), "a_new": s(self.std_new)},
                "temporal": {"a_old": s(self.tmp_old), "a_new": s(self.tmp_new)},
            },
            "prefers_old_under_standard": self.prefers_old_under_standard,
            "recovers_new_under_temporal": self.recovers_new_under_temporal,
            "is_ptc": self.is_ptc,
        }
