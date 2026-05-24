"""End-to-end TAS pipeline: detect → (if c > τ) steer.

This is the inference-time wrapper called per query. It

  1. captures h_ℓ*(q) under the standard prompt (one forward pass, hooked)
  2. computes c = detector.score(h)
  3. if c > τ, installs the steering hook for the SAME query, runs a second
     scoring forward pass with the perturbation +α·c·Δ applied at last token
  4. if c ≤ τ, returns the baseline scores (Theorem 3: no-op preservation)

The whole thing is dressed as `tas_score_candidate(prefix, candidate, ...)`
which returns the length-normalized log-prob the model assigns to the
candidate under whichever path the detector triggered.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from temporal_conflict.models import LoadedModel
from temporal_conflict.steering.detector import TrainedDetector
from temporal_conflict.steering.hooks import decoder_layer, make_capture_hook, make_steer_hook
from temporal_conflict.steering.steer import SteeringVectors


@dataclass
class TASConfig:
    layer: int
    alpha: float
    tau: float
    variant: str = "global"  # global | relation | domain
    # The product term in the steering update: + α · c · Δ.
    # If `scale_by_score=False`, we use α directly (i.e. detector is treated
    # as a gate only, not a confidence weight). The proposal's Eq. (TAS) uses
    # α · c · Δ; this flag exposes both for ablation.
    scale_by_score: bool = True


@dataclass
class TASRoute:
    """Diagnostic record for one query."""

    detector_score: float
    steered: bool
    effective_alpha: float  # the α actually applied (0 if not steered)


@torch.no_grad()
def tas_score_candidates(
    loaded: LoadedModel,
    prefix: str,
    candidates: list[str],
    detector: TrainedDetector,
    vecs: SteeringVectors,
    cfg: TASConfig,
    direction: torch.Tensor | None = None,
    candidate_prefix: str = " ",
) -> tuple[list[tuple[float, int]], TASRoute]:
    """Return [(sum_logprob, n_tokens) for each candidate] under TAS, plus the
    routing decision (so the caller can record per-query telemetry).

    Implementation notes
    --------------------
    - We need h_ℓ*(q) at the LAST PROMPT TOKEN under the standard prompt to
      score the detector. The candidate length varies per candidate, so we
      compute h once on the *prefix only* and reuse it.
    - For the scoring forward passes (one per candidate), the steering hook
      adds the perturbation at the SAME prompt-final position used by Phase
      2D, which is `prefix_len - 1` in the full [prefix+candidate] sequence.
    - `direction` overrides the vector picked from `vecs` (used by callers
      who already selected per-instance Δ).
    """
    tok = loaded.tokenizer
    device = loaded.device

    # 1. Detector forward: prefix only.
    prefix_ids = tok(prefix, return_tensors="pt",
                     add_special_tokens=True).input_ids.to(device)
    prefix_len = prefix_ids.shape[1]
    last_pos = prefix_len - 1

    layer_mod = decoder_layer(loaded.model, cfg.layer)
    buf: list[torch.Tensor] = []
    handle = layer_mod.register_forward_hook(
        make_capture_hook(buf, position=last_pos)
    )
    try:
        _ = loaded.model(prefix_ids)
    finally:
        handle.remove()
    h = buf[0].squeeze(0)  # [d_model]
    c = detector.score(h)

    steered = c > cfg.tau
    eff_alpha = (cfg.alpha * c if cfg.scale_by_score else cfg.alpha) if steered else 0.0

    # 2. Scoring forward passes for each candidate (with steering hook if applicable).
    out: list[tuple[float, int]] = []
    if steered:
        if direction is None:
            # SteeringVectors.select needs a relation/domain. If neither is
            # passed via direction, default to global.
            direction = vecs.select(variant="global")
    for cand in candidates:
        cand_text = candidate_prefix + cand
        cand_ids = tok(cand_text, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(device)
        cand_len = cand_ids.shape[1]
        full_ids = torch.cat([prefix_ids, cand_ids], dim=1)

        handle = None
        if steered:
            steer_hook = make_steer_hook(direction, alpha=eff_alpha, position=last_pos)
            handle = layer_mod.register_forward_hook(steer_hook)
        try:
            logits = loaded.model(full_ids).logits
        finally:
            if handle is not None:
                handle.remove()

        slice_start = prefix_len - 1
        slice_end = prefix_len + cand_len - 1
        cand_logits = logits[0, slice_start:slice_end, :]
        log_probs = F.log_softmax(cand_logits.float(), dim=-1)
        token_lp = log_probs.gather(1, cand_ids[0].unsqueeze(-1)).squeeze(-1)
        out.append((token_lp.sum().item(), cand_len))

    return out, TASRoute(detector_score=c, steered=steered, effective_alpha=eff_alpha)
