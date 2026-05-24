"""Stage 2: locate the conflict-critical layer.

Activation-patching protocol:

    For each verified PTC instance and each layer ℓ:
      1. Forward-pass `prompt_temporal + answer_suffix`, capture the
         residual-stream activation at the last token, at layer ℓ. Call this
         the donor h_ℓ(q, c_temp).
      2. Forward-pass `prompt_standard + answer_suffix + " " + a_old` and
         `prompt_standard + answer_suffix + " " + a_new`, but at layer ℓ
         install a forward hook that overwrites
         h_ℓ[:, last_prompt_pos, :] with the donor.
      3. Score length-normalized log P(a_old | q, patch_at_ℓ) and
         log P(a_new | q, patch_at_ℓ).
      4. The patch "flips" the answer if a_new now wins under the standard
         prompt at layer ℓ.

We define:

    AFR(ℓ) = mean over PTC instances of 1[ patched_new_lp > patched_old_lp ]

The conflict-critical layer is

    ℓ* = argmax_ℓ AFR(ℓ).

We also report per-layer mean log-probability changes for a_new (a continuous
analogue of AFR that is more informative when AFR is noisy at small N).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from temporal_conflict.models import LoadedModel
from temporal_conflict.schema import PTCInstance
from temporal_conflict.steering.hooks import decoder_layer, make_patch_hook


@dataclass
class PatchScores:
    """Log-prob scores under a single layer-patch experiment, per instance."""

    layer: int
    instance_id: str
    old_lp_baseline: float
    new_lp_baseline: float
    old_lp_patched: float
    new_lp_patched: float

    @property
    def flipped(self) -> bool:
        return self.new_lp_patched > self.old_lp_patched

    @property
    def delta_new(self) -> float:
        return self.new_lp_patched - self.new_lp_baseline


@torch.no_grad()
def _score_candidate_with_hook(
    loaded: LoadedModel,
    prefix: str,
    candidate: str,
    hook_fn=None,
    hook_module=None,
    candidate_prefix: str = " ",
) -> tuple[float, int]:
    """Length-normalized score of `candidate` given `prefix`, with optional
    forward hook installed on `hook_module`. Returns (sum_logprob, n_tokens).

    Same scoring math as ptc.models.LoadedModel.score_candidate; reproduced
    here so we can wrap the forward call in an arbitrary hook context.
    """
    tok = loaded.tokenizer
    device = loaded.device

    prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    cand_text = candidate_prefix + candidate
    cand_ids = tok(cand_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    prefix_len = prefix_ids.shape[1]
    cand_len = cand_ids.shape[1]
    full_ids = torch.cat([prefix_ids, cand_ids], dim=1)

    handle = None
    try:
        if hook_fn is not None:
            handle = hook_module.register_forward_hook(hook_fn)
        logits = loaded.model(full_ids).logits
    finally:
        if handle is not None:
            handle.remove()

    slice_start = prefix_len - 1
    slice_end = prefix_len + cand_len - 1
    cand_logits = logits[0, slice_start:slice_end, :]
    log_probs = F.log_softmax(cand_logits.float(), dim=-1)
    token_lp = log_probs.gather(1, cand_ids[0].unsqueeze(-1)).squeeze(-1)
    return token_lp.sum().item(), cand_len


def _mean_lp(sum_lp: float, n: int) -> float:
    return sum_lp / n if n > 0 else float("-inf")


@torch.no_grad()
def patch_one_instance_one_layer(
    loaded: LoadedModel,
    instance: PTCInstance,
    layer_idx: int,
    donor_vec: torch.Tensor,
    answer_suffix: str = "\nAnswer:",
    candidate_prefix: str = " ",
) -> PatchScores:
    """Run two scoring passes under standard prompt, with a hook that
    overwrites the residual-stream activation at `layer_idx`, last token,
    with `donor_vec` (from the temporal forward pass at the same layer).
    Also returns the unpatched baseline scores for the same prompt.
    """
    std_prefix = instance.prompt_standard + answer_suffix
    std_prefix_ids = loaded.tokenizer(
        std_prefix, return_tensors="pt", add_special_tokens=True,
    ).input_ids
    last_pos = std_prefix_ids.shape[1] - 1

    layer_mod = decoder_layer(loaded.model, layer_idx)

    # Baseline (no hook).
    old_b_sum, old_b_n = _score_candidate_with_hook(
        loaded, std_prefix, instance.a_old_label, candidate_prefix=candidate_prefix,
    )
    new_b_sum, new_b_n = _score_candidate_with_hook(
        loaded, std_prefix, instance.a_new_label, candidate_prefix=candidate_prefix,
    )

    # Patched: install hook for both candidate scoring passes.
    hook_fn = make_patch_hook(donor_vec, position=last_pos)
    old_p_sum, old_p_n = _score_candidate_with_hook(
        loaded, std_prefix, instance.a_old_label,
        hook_fn=hook_fn, hook_module=layer_mod, candidate_prefix=candidate_prefix,
    )
    new_p_sum, new_p_n = _score_candidate_with_hook(
        loaded, std_prefix, instance.a_new_label,
        hook_fn=hook_fn, hook_module=layer_mod, candidate_prefix=candidate_prefix,
    )

    return PatchScores(
        layer=layer_idx,
        instance_id=instance.instance_id,
        old_lp_baseline=_mean_lp(old_b_sum, old_b_n),
        new_lp_baseline=_mean_lp(new_b_sum, new_b_n),
        old_lp_patched=_mean_lp(old_p_sum, old_p_n),
        new_lp_patched=_mean_lp(new_p_sum, new_p_n),
    )


def layer_sweep(
    loaded: LoadedModel,
    instances: list[PTCInstance],
    cached_temporal_activations: torch.Tensor,
    layers: list[int],
    answer_suffix: str = "\nAnswer:",
    candidate_prefix: str = " ",
    log_every: int = 5,
) -> dict[int, list[PatchScores]]:
    """Run the activation-patching sweep across `layers`.

    `cached_temporal_activations` is shape [N, L, d_model] from
    ptc.steering.activations.save_activations / load_activations
    (the 'temporal' tensor), aligned with `instances` by index.
    """
    assert cached_temporal_activations.shape[0] == len(instances), (
        f"activation cache N={cached_temporal_activations.shape[0]} != "
        f"#instances {len(instances)}"
    )

    out: dict[int, list[PatchScores]] = {}
    for li, layer_idx in enumerate(layers):
        per_inst: list[PatchScores] = []
        for i, inst in enumerate(instances):
            donor = cached_temporal_activations[i, layer_idx, :]
            ps = patch_one_instance_one_layer(
                loaded, inst, layer_idx, donor,
                answer_suffix=answer_suffix, candidate_prefix=candidate_prefix,
            )
            per_inst.append(ps)
        out[layer_idx] = per_inst
        if (li + 1) % log_every == 0 or li == len(layers) - 1:
            afr = sum(p.flipped for p in per_inst) / len(per_inst)
            d_new = sum(p.delta_new for p in per_inst) / len(per_inst)
            print(f"  layer {layer_idx:>3}  AFR={afr:.3f}  mean Δlp(a_new)={d_new:+.3f}",
                  flush=True)
    return out


def summarize_sweep(per_layer: dict[int, list[PatchScores]]) -> dict[int, dict]:
    """Compute AFR(ℓ) and mean log-prob deltas per layer."""
    summary: dict[int, dict] = {}
    for layer_idx, scores in per_layer.items():
        n = len(scores)
        afr = sum(p.flipped for p in scores) / n
        d_new = sum(p.delta_new for p in scores) / n
        d_old = sum(p.old_lp_patched - p.old_lp_baseline for p in scores) / n
        summary[layer_idx] = {
            "n": n,
            "afr": afr,
            "mean_delta_new": d_new,
            "mean_delta_old": d_old,
        }
    return summary


def critical_layer(summary: dict[int, dict]) -> int:
    """ℓ* = argmax AFR; ties broken by larger mean_delta_new."""
    items = sorted(
        summary.items(),
        key=lambda kv: (kv[1]["afr"], kv[1]["mean_delta_new"]),
        reverse=True,
    )
    return items[0][0]
