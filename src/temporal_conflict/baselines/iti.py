"""Reproducible ITI baseline direction (Bucket 5, Part D).

Matches the supplement's ITI definition exactly:

    Delta_ITI = mean(h_l*(q) | theta prefers a_new under standard prompt)
              - mean(h_l*(q) | theta prefers a_old, verified-PTC),

using standard-prompt hidden states at the conflict-critical layer l*, the same
last-prompt-token position and residual hook as TAS. Only the *construction* of
the direction differs from TAS; l*, the alpha grid, the control set, and the J
objective are shared.

The standard-prompt hidden states are already cached (benchmark-ordered) in
``runs/tas_large/<model>/detector_activations.pt`` (``H`` at l*), so the ITI
direction is built with NO model forward passes. Construction uses TRAIN records
only; each side is capped at 500 records (deterministic seed 0), matching the
stated protocol.

Direction scaling: a raw mean-difference vector (NOT unit-normalized, NOT
norm-matched to TAS). The shared alpha grid explores scale independently for
each method, so only Delta construction differs. The vector norm is recorded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ITIDirection:
    layer: int
    delta: np.ndarray                 # [d_model], raw mean-difference at l*
    n_positive: int
    n_negative: int
    cap: int
    seed: int
    norm: float
    positive_record_ids: list[str] = field(default_factory=list)
    negative_record_ids: list[str] = field(default_factory=list)
    normalization: str = "raw_mean_difference"


def _sample(ids: list[int], cap: int, seed: int, tag: str) -> list[int]:
    """Deterministic subsample of row indices, capped at `cap`."""
    ids = sorted(ids)
    if len(ids) <= cap:
        return ids
    rng = random.Random(f"{seed}:{tag}")
    return sorted(rng.sample(ids, cap))


def build_iti_direction(
    H: np.ndarray,                    # [N, d_model] standard-prompt h at l*
    layer: int,
    train_rows: list[int],            # benchmark row indices in TRAIN split
    prefers_new: np.ndarray,          # bool [N]: standard argmax == a_new
    is_ptc: np.ndarray,               # bool [N]: verified PTC (prefers-old)
    record_id_of: dict[int, str],     # row index -> record_id
    cap: int = 500,
    seed: int = 0,
) -> ITIDirection:
    """Delta_ITI from TRAIN rows only. Positive = prefers-new; negative =
    verified-PTC (prefers-old). Both restricted to train and capped."""
    train = set(train_rows)
    pos_rows = [i for i in train_rows if prefers_new[i]]
    neg_rows = [i for i in train_rows if is_ptc[i]]
    if not pos_rows or not neg_rows:
        raise ValueError(
            f"ITI needs both groups on train: pos={len(pos_rows)} neg={len(neg_rows)}")
    # Positive and negative groups must be train-disjoint from each other by
    # construction (prefers-new vs prefers-old are mutually exclusive argmax).
    assert not (set(pos_rows) & set(neg_rows)), "ITI pos/neg overlap"

    pos_rows = _sample(pos_rows, cap, seed, "pos")
    neg_rows = _sample(neg_rows, cap, seed, "neg")
    assert train.issuperset(pos_rows) and train.issuperset(neg_rows), \
        "ITI groups leaked off-train"

    mu_pos = H[np.array(pos_rows)].mean(axis=0)
    mu_neg = H[np.array(neg_rows)].mean(axis=0)
    delta = (mu_pos - mu_neg).astype(np.float32)

    return ITIDirection(
        layer=layer, delta=delta,
        n_positive=len(pos_rows), n_negative=len(neg_rows),
        cap=cap, seed=seed, norm=float(np.linalg.norm(delta)),
        positive_record_ids=[record_id_of[i] for i in pos_rows],
        negative_record_ids=[record_id_of[i] for i in neg_rows],
    )
