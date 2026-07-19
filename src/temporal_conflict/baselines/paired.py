"""Paired held-out TAS-vs-ITI evaluation core (Bucket 5).

Shared, model-driven helpers used by ``scripts/run_paired_tas_iti.py``:
direction construction (train-only), a validation alpha sweep, and a single
held-out test evaluation, all on the authoritative subject-disjoint split.

Baseline (pre-edit, standard-prompt) scores come from the cached, benchmark-
ordered screening file, so only the *steered* (post-edit) scores require
forward passes. Only Delta construction differs between the methods; l*, the
alpha grid, the control set, the scoring, and the J objective are shared.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from temporal_conflict.models import LoadedModel
from temporal_conflict.steering.hooks import decoder_layer, make_capture_hook
from temporal_conflict.steering.steer import score_with_steering

ALPHA_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)


@dataclass
class Record:
    row_index: int
    record_id: str
    subject_qid: str
    relation_id: str
    update_date: str
    prompt_standard: str
    prompt_temporal: str
    a_old_label: str
    a_new_label: str
    # cached baseline (standard prompt) length-normalized log-probs
    std_old: float
    std_new: float
    is_ptc: bool
    knowledge_absent: bool


@torch.no_grad()
def capture_hidden(loaded: LoadedModel, prompt: str, layer: int) -> np.ndarray:
    """Residual-stream vector at `layer`, last prompt token."""
    tok = loaded.tokenizer
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True
              ).input_ids.to(loaded.device)
    last = ids.shape[1] - 1
    buf: list[torch.Tensor] = []
    h = decoder_layer(loaded.model, layer).register_forward_hook(
        make_capture_hook(buf, position=last))
    try:
        loaded.model(ids)
    finally:
        h.remove()
    return buf[0].squeeze(0).detach().cpu().float().numpy()


def build_tas_v2_trainonly(
    loaded: LoadedModel, train_ptc: list[Record], layer: int,
    H_standard: np.ndarray,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    """V2 per-relation Delta_TAS from TRAIN filtered-PTC records only.

    Delta_rel = mean_{q in rel} [ h_l*(q, c_temp) - h_l*(q) ], where the
    standard hidden state h_l*(q) is read from the cached ``H_standard`` and the
    temporal hidden state is computed fresh (forward pass on the temporal
    prompt). Returns (per_relation dict, global fallback, metadata).
    """
    by_rel: dict[str, list[np.ndarray]] = defaultdict(list)
    all_diffs: list[np.ndarray] = []
    used_ids: list[str] = []
    for rec in train_ptc:
        h_std = H_standard[rec.row_index]
        h_tmp = capture_hidden(loaded, rec.prompt_temporal, layer)
        diff = h_tmp - h_std
        by_rel[rec.relation_id].append(diff)
        all_diffs.append(diff)
        used_ids.append(rec.record_id)
    global_delta = torch.tensor(np.mean(all_diffs, axis=0), dtype=torch.float32)
    per_relation = {rel: torch.tensor(np.mean(v, axis=0), dtype=torch.float32)
                    for rel, v in by_rel.items() if len(v) >= 3}
    meta = {
        "n_used": len(used_ids),
        "per_relation_counts": {r: len(v) for r, v in by_rel.items()},
        "per_relation_kept": sorted(per_relation),
        "global_norm": float(global_delta.norm()),
        "train_record_ids": used_ids,
    }
    return per_relation, global_delta, meta


def tas_direction_for(rec: Record, per_relation: dict[str, torch.Tensor],
                      global_delta: torch.Tensor) -> torch.Tensor:
    return per_relation.get(rec.relation_id, global_delta)


@torch.no_grad()
def steered_old_new(loaded: LoadedModel, rec: Record, layer: int,
                    direction: torch.Tensor, alpha: float,
                    answer_suffix: str, candidate_prefix: str) -> tuple[float, float]:
    prefix = rec.prompt_standard + answer_suffix
    o_s, o_n = score_with_steering(loaded, prefix, rec.a_old_label, layer,
                                   direction, alpha, candidate_prefix=candidate_prefix)
    n_s, n_n = score_with_steering(loaded, prefix, rec.a_new_label, layer,
                                   direction, alpha, candidate_prefix=candidate_prefix)
    return o_s / max(o_n, 1), n_s / max(n_n, 1)


def evaluate(loaded: LoadedModel, ptc: list[Record], controls: list[Record],
             layer: int, alpha: float, direction_fn,
             answer_suffix: str, candidate_prefix: str) -> dict:
    """Recovery on PTC + PA on controls at one alpha. Returns summary + rows."""
    rows = []
    rec_hits = 0
    for rec in ptc:
        d = direction_fn(rec)
        o, n = steered_old_new(loaded, rec, layer, d, alpha,
                               answer_suffix, candidate_prefix)
        recovered = int(n > o)
        rec_hits += recovered
        rows.append({"record_id": rec.record_id, "subject_qid": rec.subject_qid,
                     "relation_id": rec.relation_id, "update_date": rec.update_date,
                     "subset": "ptc", "standard_old_score": rec.std_old,
                     "standard_new_score": rec.std_new,
                     "post_edit_old_score": o, "post_edit_new_score": n,
                     "pre_edit_winner": "new" if rec.std_new > rec.std_old else "old",
                     "post_edit_winner": "new" if n > o else "old",
                     "recovery_indicator": recovered,
                     "direction_norm": float(d.norm())})
    pa_hits = 0
    for rec in controls:
        d = direction_fn(rec)
        o, n = steered_old_new(loaded, rec, layer, d, alpha,
                               answer_suffix, candidate_prefix)
        base_new = rec.std_new > rec.std_old
        post_new = n > o
        preserved = int(post_new == base_new)
        pa_hits += preserved
        rows.append({"record_id": rec.record_id, "subject_qid": rec.subject_qid,
                     "relation_id": rec.relation_id, "update_date": rec.update_date,
                     "subset": "control", "standard_old_score": rec.std_old,
                     "standard_new_score": rec.std_new,
                     "post_edit_old_score": o, "post_edit_new_score": n,
                     "pre_edit_winner": "new" if base_new else "old",
                     "post_edit_winner": "new" if post_new else "old",
                     "preservation_indicator": preserved,
                     "direction_norm": float(d.norm())})
    recovery = rec_hits / len(ptc) if ptc else 0.0
    pa = pa_hits / len(controls) if controls else 0.0
    return {"alpha": alpha, "n_ptc": len(ptc), "n_control": len(controls),
            "recovery": recovery, "pa": pa, "J": recovery - (1 - pa),
            "rows": rows}


def select_alpha(sweep: list[dict]) -> dict:
    """Argmax J on validation; tie -> smaller alpha (deterministic)."""
    best = max(sweep, key=lambda s: (round(s["J"], 6), -s["alpha"]))
    ties = [s["alpha"] for s in sweep if round(s["J"], 6) == round(best["J"], 6)]
    return {"alpha": best["alpha"], "val_recovery": best["recovery"],
            "val_pa": best["pa"], "val_J": best["J"],
            "tie_break": {"tied_alphas": sorted(ties),
                          "rule": "smaller-alpha-wins" if len(ties) > 1 else "unique"}}
