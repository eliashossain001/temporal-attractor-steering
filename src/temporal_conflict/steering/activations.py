"""High-level activation extraction for PTC instances.

For each instance and each layer, captures the residual-stream activation at
the **last prompt token** under two conditions:

    standard:   prompt_standard + answer_suffix
    temporal:   prompt_temporal + answer_suffix

The "last prompt token" is the position whose next-token prediction is the
first token of the answer. This is the same position the Phase 1 scorer
attributes the answer probability to, so the captured activation is the
representation that decides the answer.

The returned tensors are dense [num_layers, d_model] per (instance, condition),
which makes Stage 2 (locator) and Stage 3 (steering-vector construction)
trivial array ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from temporal_conflict.models import LoadedModel
from temporal_conflict.schema import PTCInstance
from temporal_conflict.steering.hooks import decoder_layer, hooked, make_capture_hook, num_layers


@dataclass
class ActivationRecord:
    """Per-instance activations under both prompt conditions.

    `standard` and `temporal` are each shape [L, d_model] where L = num decoder
    layers. `last_pos_standard` and `last_pos_temporal` record the token
    indices that were captured (useful for Stage 2 patching, which has to
    match positions).
    """

    instance_id: str
    relation_pid: str
    domain: str
    standard: torch.Tensor  # [L, d_model]
    temporal: torch.Tensor  # [L, d_model]
    last_pos_standard: int
    last_pos_temporal: int


@torch.no_grad()
def _forward_and_capture(
    loaded: LoadedModel,
    text: str,
) -> tuple[list[torch.Tensor], int]:
    """Single forward pass over `text` with a capture hook on every decoder
    layer. Returns (per-layer last-token activations, last_pos).
    """
    tok = loaded.tokenizer
    ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids.to(
        loaded.device
    )
    last_pos = ids.shape[1] - 1

    L = num_layers(loaded.model)
    buffers: list[list[torch.Tensor]] = [[] for _ in range(L)]
    handles = []
    try:
        for i in range(L):
            handle = decoder_layer(loaded.model, i).register_forward_hook(
                make_capture_hook(buffers[i], position=last_pos)
            )
            handles.append(handle)
        _ = loaded.model(ids)
    finally:
        for h in handles:
            h.remove()

    # Each buffer holds one tensor of shape [1, d_model]; stack to [L, d_model].
    per_layer = torch.stack([b[0].squeeze(0).cpu() for b in buffers], dim=0)
    return per_layer, last_pos


def extract_activations(
    loaded: LoadedModel,
    instance: PTCInstance,
    answer_suffix: str = "\nAnswer:",
) -> ActivationRecord:
    """Run two forward passes (standard + temporal) and capture per-layer
    last-token activations.
    """
    std_text = instance.prompt_standard + answer_suffix
    tmp_text = instance.prompt_temporal + answer_suffix

    std_acts, std_pos = _forward_and_capture(loaded, std_text)
    tmp_acts, tmp_pos = _forward_and_capture(loaded, tmp_text)

    return ActivationRecord(
        instance_id=instance.instance_id,
        relation_pid=instance.relation_pid,
        domain=instance.domain,
        standard=std_acts,
        temporal=tmp_acts,
        last_pos_standard=std_pos,
        last_pos_temporal=tmp_pos,
    )


def extract_many(
    loaded: LoadedModel,
    instances: Iterable[PTCInstance],
    answer_suffix: str = "\nAnswer:",
    log_every: int = 25,
) -> list[ActivationRecord]:
    """Convenience wrapper. Returns a list aligned with `instances` order."""
    out: list[ActivationRecord] = []
    instances = list(instances)
    for i, inst in enumerate(instances):
        out.append(extract_activations(loaded, inst, answer_suffix))
        if (i + 1) % log_every == 0:
            print(f"  cached activations {i + 1}/{len(instances)}", flush=True)
    return out


def save_activations(path: str, records: list[ActivationRecord]) -> None:
    """Serialize to a single torch file for fast reload."""
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "instance_ids": [r.instance_id for r in records],
        "relation_pids": [r.relation_pid for r in records],
        "domains": [r.domain for r in records],
        "standard": torch.stack([r.standard for r in records], dim=0),  # [N, L, d]
        "temporal": torch.stack([r.temporal for r in records], dim=0),  # [N, L, d]
        "last_pos_standard": torch.tensor([r.last_pos_standard for r in records]),
        "last_pos_temporal": torch.tensor([r.last_pos_temporal for r in records]),
    }
    torch.save(payload, path)


def load_activations(path: str) -> dict:
    """Load the dict written by `save_activations`."""
    return torch.load(path, map_location="cpu", weights_only=False)
