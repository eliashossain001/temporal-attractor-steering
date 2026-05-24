"""Stage 3: steering vector construction and inference-time intervention.

Vector construction (V1 global):

    Δ = mean over PTC instances of [ h_ℓ*(q, c_temp) - h_ℓ*(q) ]

(captured at the last prompt token in each forward pass).

V2 (relation-specific) and V3 (domain-specific) are computed by partitioning
the PTC subset before averaging. All three are returned together so downstream
code can pick the appropriate Δ at scoring time.

Intervention (Theorem 2 / Eq. (TAS update)):

    h'_ℓ*[:, last_pos, :] = h_ℓ*[:, last_pos, :] + α · c · Δ

When the detector is the oracle (always 1 on the PTC subset, always 0 off it),
this collapses to + α · Δ. The steer hook is a thin wrapper around
ptc.steering.hooks.make_steer_hook installed on the ℓ*-th decoder layer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from temporal_conflict.models import LoadedModel
from temporal_conflict.schema import PTCInstance
from temporal_conflict.steering.hooks import decoder_layer, make_steer_hook


@dataclass
class SteeringVectors:
    """All Δ variants for a single model at a single layer ℓ*."""

    layer: int
    d_model: int
    n_used: int
    global_delta: torch.Tensor  # [d_model]
    per_relation: dict[str, torch.Tensor] = field(default_factory=dict)
    per_domain: dict[str, torch.Tensor] = field(default_factory=dict)

    def select(
        self,
        variant: str = "global",
        relation_pid: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> torch.Tensor:
        """Pick the right Δ for a query.

        variant ∈ {"global", "relation", "domain"}.
        Falls back to global if the per-relation/per-domain vector was not
        computed (e.g. relation not in the training subset).
        """
        if variant == "global":
            return self.global_delta
        if variant == "relation" and relation_pid in self.per_relation:
            return self.per_relation[relation_pid]
        if variant == "domain" and domain in self.per_domain:
            return self.per_domain[domain]
        return self.global_delta


def build_vectors(
    layer: int,
    standard_acts: torch.Tensor,  # [N, L, d_model]
    temporal_acts: torch.Tensor,  # [N, L, d_model]
    relation_pids: list[str],
    domains: list[str],
    compute_per_relation: bool = False,
    compute_per_domain: bool = False,
) -> SteeringVectors:
    """Construct V1/V2/V3 steering vectors from cached activations at `layer`.

    Standard and temporal activation caches must be aligned (same N).
    """
    N, L, d_model = standard_acts.shape
    assert temporal_acts.shape == standard_acts.shape, \
        f"shape mismatch: {standard_acts.shape} vs {temporal_acts.shape}"
    assert 0 <= layer < L, f"layer {layer} out of range [0, {L})"
    assert len(relation_pids) == N and len(domains) == N

    # Slice to the chosen layer once.
    std_l = standard_acts[:, layer, :].float()
    tmp_l = temporal_acts[:, layer, :].float()
    deltas = tmp_l - std_l  # [N, d_model]

    global_delta = deltas.mean(dim=0)

    per_relation: dict[str, torch.Tensor] = {}
    if compute_per_relation:
        by_rel: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(relation_pids):
            by_rel[r].append(i)
        for r, idxs in by_rel.items():
            if len(idxs) >= 3:  # require at least 3 samples for a stable mean
                per_relation[r] = deltas[idxs].mean(dim=0)

    per_domain: dict[str, torch.Tensor] = {}
    if compute_per_domain:
        by_dom: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(domains):
            by_dom[d].append(i)
        for d, idxs in by_dom.items():
            if len(idxs) >= 3:
                per_domain[d] = deltas[idxs].mean(dim=0)

    return SteeringVectors(
        layer=layer,
        d_model=d_model,
        n_used=N,
        global_delta=global_delta,
        per_relation=per_relation,
        per_domain=per_domain,
    )


def save_vectors(path: str, vecs: SteeringVectors) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "layer": vecs.layer,
        "d_model": vecs.d_model,
        "n_used": vecs.n_used,
        "global_delta": vecs.global_delta,
        "per_relation": vecs.per_relation,
        "per_domain": vecs.per_domain,
    }
    torch.save(payload, path)


def load_vectors(path: str) -> SteeringVectors:
    p = torch.load(path, map_location="cpu", weights_only=False)
    return SteeringVectors(
        layer=p["layer"],
        d_model=p["d_model"],
        n_used=p["n_used"],
        global_delta=p["global_delta"],
        per_relation=p.get("per_relation", {}),
        per_domain=p.get("per_domain", {}),
    )


@torch.no_grad()
def score_with_steering(
    loaded: LoadedModel,
    prefix: str,
    candidate: str,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    candidate_prefix: str = " ",
) -> tuple[float, int]:
    """Length-normalized log P(candidate | prefix) with steering at `layer`,
    last prompt token. Returns (sum_logprob, n_tokens).

    Mirrors ptc.steering.locate._score_candidate_with_hook but for steering.
    """
    tok = loaded.tokenizer
    device = loaded.device

    prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    cand_text = candidate_prefix + candidate
    cand_ids = tok(cand_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    prefix_len = prefix_ids.shape[1]
    cand_len = cand_ids.shape[1]
    full_ids = torch.cat([prefix_ids, cand_ids], dim=1)
    last_pos = prefix_len - 1  # last prompt token position

    layer_mod = decoder_layer(loaded.model, layer)
    hook_fn = make_steer_hook(direction, alpha=alpha, position=last_pos)
    handle = layer_mod.register_forward_hook(hook_fn)
    try:
        logits = loaded.model(full_ids).logits
    finally:
        handle.remove()

    slice_start = prefix_len - 1
    slice_end = prefix_len + cand_len - 1
    cand_logits = logits[0, slice_start:slice_end, :]
    log_probs = F.log_softmax(cand_logits.float(), dim=-1)
    token_lp = log_probs.gather(1, cand_ids[0].unsqueeze(-1)).squeeze(-1)
    return token_lp.sum().item(), cand_len
