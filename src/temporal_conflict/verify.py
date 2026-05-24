"""Apply the PTC definition (Def. 2) to a single instance.

Standard prompt:  P(a_old | q)  >  P(a_new | q)              => prefers_old
Temporal prompt:  P(a_new | q, c_temp)  >  P(a_old | q, c_temp) => recovers_new
PTC = prefers_old AND recovers_new.
"""

from __future__ import annotations

from typing import Iterable

from temporal_conflict.models import LoadedModel
from temporal_conflict.schema import InstanceResult, PTCInstance


def verify_instance(
    model: LoadedModel,
    instance: PTCInstance,
    answer_suffix: str = "\nAnswer:",
    candidate_prefix: str = " ",
) -> InstanceResult:
    std_prefix = instance.prompt_standard + answer_suffix
    tmp_prefix = instance.prompt_temporal + answer_suffix

    std_old = model.score_candidate(std_prefix, instance.a_old_label, candidate_prefix)
    std_new = model.score_candidate(std_prefix, instance.a_new_label, candidate_prefix)
    tmp_old = model.score_candidate(tmp_prefix, instance.a_old_label, candidate_prefix)
    tmp_new = model.score_candidate(tmp_prefix, instance.a_new_label, candidate_prefix)

    prefers_old = std_old.mean_logprob > std_new.mean_logprob
    recovers_new = tmp_new.mean_logprob > tmp_old.mean_logprob

    return InstanceResult(
        instance_id=instance.instance_id,
        model_name=model.name,
        relation_pid=instance.relation_pid,
        domain=instance.domain,
        std_old=std_old,
        std_new=std_new,
        tmp_old=tmp_old,
        tmp_new=tmp_new,
        prefers_old_under_standard=prefers_old,
        recovers_new_under_temporal=recovers_new,
        is_ptc=prefers_old and recovers_new,
    )


def verify_instances(
    model: LoadedModel,
    instances: Iterable[PTCInstance],
    answer_suffix: str = "\nAnswer:",
    candidate_prefix: str = " ",
) -> list[InstanceResult]:
    return [
        verify_instance(model, inst, answer_suffix, candidate_prefix)
        for inst in instances
    ]
