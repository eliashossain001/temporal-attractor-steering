"""WP-A: prompt-based baselines (checklist item 1).

The reviewer's central request is an honest comparison against cheap prompting
interventions before claiming a mechanistic method is needed. We evaluate three
prompt families on the *same* verified-PTC subset and non-conflict control set
that TAS is scored on, using the identical length-normalised log-prob scoring,
so the only thing that varies is the prompt text:

  * ``standard``     -- the bare question (this is the PTC default; Recovery ~ 0
                        by construction, included as the floor).
  * ``date_prefix``  -- a minimal date cue prepended to the question, without the
                        "as of ... who was" reformulation.
  * ``instruction``  -- an instruction to answer with the current/latest fact.
  * ``temporal``     -- the full temporal cue used to *define* PTC. It is
                        circular on the PTC subset (Recovery = 1 by construction)
                        and is reported only as the ceiling / sanity anchor.

For each prompt we report Recovery on the PTC subset and Preservation Accuracy
(argmax unchanged vs. the standard prompt) on the non-conflict control set, and
we emit per-instance decisions so the paired McNemar tests in
``analysis.stats`` can compare TAS against each prompt baseline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from temporal_conflict.models import LoadedModel
from temporal_conflict.schema import PTCInstance
from temporal_conflict.steering.locate import _mean_lp, _score_candidate_with_hook

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _instance_year(inst: PTCInstance) -> str:
    """The temporal year to use for the date cue.

    Prefer the year already embedded in the verified temporal prompt (keeps the
    date cue and the definitional temporal prompt anchored to the same year);
    fall back to the update date, then the new-fact start.
    """
    for source in (inst.prompt_temporal, inst.t_update or "", inst.t_new_start or ""):
        m = _YEAR_RE.search(source or "")
        if m:
            return m.group(0)
    return "2024"


# Prompt builders: (instance) -> prefix string (before the answer suffix).
PROMPT_BUILDERS: dict[str, Callable[[PTCInstance], str]] = {
    "standard": lambda i: i.prompt_standard,
    "temporal": lambda i: i.prompt_temporal,
    "date_prefix": lambda i: f"In {_instance_year(i)}: {i.prompt_standard}",
    "instruction": lambda i: (
        "Answer with the current, most up-to-date fact. " + i.prompt_standard
    ),
}

# The non-trivial baselines we actually run a model for (standard/temporal are
# already available from the screening pass and need no new forward passes).
NEW_PROMPTS = ("date_prefix", "instruction")


@dataclass
class PromptBaselineConfig:
    prompts: tuple[str, ...] = NEW_PROMPTS
    answer_suffix: str = "\nAnswer:"
    candidate_prefix: str = " "


@torch.no_grad()
def _score_old_new(
    loaded: LoadedModel, prefix: str, inst: PTCInstance, cfg: PromptBaselineConfig
) -> tuple[float, float]:
    """Length-normalised (old, new) log-probs under ``prefix`` (no intervention)."""
    text = prefix + cfg.answer_suffix
    old_sum, old_n = _score_candidate_with_hook(
        loaded, text, inst.a_old_label, candidate_prefix=cfg.candidate_prefix
    )
    new_sum, new_n = _score_candidate_with_hook(
        loaded, text, inst.a_new_label, candidate_prefix=cfg.candidate_prefix
    )
    return _mean_lp(old_sum, old_n), _mean_lp(new_sum, new_n)


def evaluate_prompt_baselines(
    loaded: LoadedModel,
    ptc_instances: list[PTCInstance],
    control_instances: list[PTCInstance],
    screening_by_id: dict[str, dict],
    cfg: PromptBaselineConfig,
) -> dict:
    """Run each configured prompt family and return a results dict.

    ``screening_by_id`` supplies the cached standard-prompt argmax used as the
    preservation reference (so PA is defined identically to the TAS evaluation).
    """
    prompts = list(cfg.prompts)
    out_per_instance: dict[str, list[dict]] = {p: [] for p in prompts}

    for name in prompts:
        builder = PROMPT_BUILDERS[name]
        # Recovery on the verified PTC subset.
        for inst in ptc_instances:
            old_lp, new_lp = _score_old_new(loaded, builder(inst), inst, cfg)
            out_per_instance[name].append({
                "instance_id": inst.instance_id,
                "relation_pid": inst.relation_pid,
                "domain": inst.domain,
                "subset": "ptc",
                "old_lp": old_lp,
                "new_lp": new_lp,
                "recovered": int(new_lp > old_lp),
            })
        # Preservation on the non-conflict control subset.
        for inst in control_instances:
            old_lp, new_lp = _score_old_new(loaded, builder(inst), inst, cfg)
            r = screening_by_id.get(inst.instance_id, {})
            std = r.get("scores", {}).get("standard", {})
            base_new = std.get("a_new", {}).get("mean_logprob")
            base_old = std.get("a_old", {}).get("mean_logprob")
            if base_new is None or base_old is None:
                continue
            base_top_new = base_new > base_old
            alt_top_new = new_lp > old_lp
            out_per_instance[name].append({
                "instance_id": inst.instance_id,
                "relation_pid": inst.relation_pid,
                "domain": inst.domain,
                "subset": "control",
                "old_lp": old_lp,
                "new_lp": new_lp,
                "preserved": int(alt_top_new == base_top_new),
            })

    summary = {}
    for name in prompts:
        rows = out_per_instance[name]
        ptc = [r for r in rows if r["subset"] == "ptc"]
        ctrl = [r for r in rows if r["subset"] == "control"]
        summary[name] = {
            "n_ptc": len(ptc),
            "recovery": (sum(r["recovered"] for r in ptc) / len(ptc)) if ptc else 0.0,
            "n_control": len(ctrl),
            "preservation_accuracy": (
                sum(r["preserved"] for r in ctrl) / len(ctrl) if ctrl else 0.0
            ),
        }

    return {
        "model": loaded.name,
        "config": {
            "prompts": prompts,
            "answer_suffix": cfg.answer_suffix,
            "candidate_prefix": cfg.candidate_prefix,
        },
        "summary": summary,
        "per_instance": out_per_instance,
    }


def write_results(path: str | Path, results: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2))
