"""HF forward-hook utilities for capturing and intervening on residual-stream
activations.

Three hook factories cover the three operations TAS needs:

    make_capture_hook   -- write hidden[:, position, :] into a list (Stage 2 input,
                           Stage 3 vector construction).
    make_patch_hook     -- replace hidden[:, position, :] with a donor vector
                           (Stage 2 activation patching).
    make_steer_hook     -- add alpha * vec into hidden[:, position, :]
                           (Stage 3 inference-time intervention).

All hooks handle the LlamaDecoderLayer / MistralDecoderLayer / GemmaDecoderLayer
output convention, which is a tuple `(hidden_states, ...)`. For plain modules
that return a tensor (e.g. MLP / attention sublayers) they pass through.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn as nn


def _unpack(output):
    """Return (hidden_tensor, rest_of_tuple_or_None)."""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _repack(hidden, rest):
    if rest is None:
        return hidden
    return (hidden,) + rest


def make_capture_hook(
    buffer: list[torch.Tensor],
    position: int = -1,
) -> Callable:
    """Hook that appends hidden_states[:, position, :] to `buffer` and returns
    the original output unchanged.
    """

    def hook(module, inputs, output):
        hidden, _ = _unpack(output)
        buffer.append(hidden[:, position, :].detach().clone())
        return output

    return hook


def make_patch_hook(
    donor: torch.Tensor,
    position: int = -1,
) -> Callable:
    """Hook that replaces hidden_states[:, position, :] with `donor`
    (a [d_model] tensor, broadcast across batch).
    """

    def hook(module, inputs, output):
        hidden, rest = _unpack(output)
        new_hidden = hidden.clone()
        new_hidden[:, position, :] = donor.to(
            dtype=new_hidden.dtype, device=new_hidden.device
        )
        return _repack(new_hidden, rest)

    return hook


def make_steer_hook(
    direction: torch.Tensor,
    alpha: float,
    position: int = -1,
) -> Callable:
    """Hook that adds `alpha * direction` to hidden_states[:, position, :]."""

    def hook(module, inputs, output):
        hidden, rest = _unpack(output)
        new_hidden = hidden.clone()
        delta = (alpha * direction).to(dtype=new_hidden.dtype, device=new_hidden.device)
        new_hidden[:, position, :] = new_hidden[:, position, :] + delta
        return _repack(new_hidden, rest)

    return hook


@contextmanager
def hooked(module: nn.Module, hook_fn: Callable):
    """Context manager that registers a forward hook and removes it on exit."""
    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def decoder_layer(model, layer_idx: int) -> nn.Module:
    """Locate the decoder-layer module for `layer_idx` across common LM
    architectures (Llama / Mistral / Qwen / Gemma).

    All of these expose `model.model.layers[i]`; Pythia uses
    `model.gpt_neox.layers[i]`. Extend here if a new family is added.
    """
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return base.layers[layer_idx]
    neox = getattr(model, "gpt_neox", None)
    if neox is not None and hasattr(neox, "layers"):
        return neox.layers[layer_idx]
    raise AttributeError(
        f"Could not locate decoder layers on {type(model).__name__}. "
        f"Extend ptc.steering.hooks.decoder_layer for this family."
    )


def num_layers(model) -> int:
    """Number of decoder layers (depth L)."""
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return len(base.layers)
    neox = getattr(model, "gpt_neox", None)
    if neox is not None and hasattr(neox, "layers"):
        return len(neox.layers)
    cfg = model.config
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise AttributeError(f"Could not determine layer count for {type(model).__name__}")


def layer_sublayers(layer: nn.Module) -> dict[str, nn.Module]:
    """Return a dict mapping {"attn": attention_module, "mlp": mlp_module}
    if both are present on the decoder layer, else empty dict for sublayer
    decomposition during locator sweeps.
    """
    out: dict[str, nn.Module] = {}
    for attr in ("self_attn", "attention", "attn"):
        if hasattr(layer, attr):
            out["attn"] = getattr(layer, attr)
            break
    for attr in ("mlp", "feed_forward"):
        if hasattr(layer, attr):
            out["mlp"] = getattr(layer, attr)
            break
    return out
