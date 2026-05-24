"""HF model loading and candidate-conditional log-probability scoring.

The scorer is the workhorse of Phase 1: given a `prefix` (the question plus
an answer cue) and a `candidate` (the verified old or new answer string),
it returns sum and count of token log-probabilities under the model. We
length-normalize downstream so candidates of different token lengths can
be compared fairly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from temporal_conflict.schema import ScoreRecord


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class LoadedModel:
    name: str
    hf_id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device

    @torch.no_grad()
    def score_candidate(
        self,
        prefix: str,
        candidate: str,
        candidate_prefix: str = " ",
    ) -> ScoreRecord:
        """Return log P(candidate | prefix) summed over candidate tokens.

        The candidate is prepended with `candidate_prefix` (default: a single
        space) so that BPE/SentencePiece tokenizers start a fresh word
        boundary. This keeps tokenization stable regardless of how `prefix`
        terminates.
        """
        tok = self.tokenizer
        device = self.device

        prefix_ids = tok(
            prefix, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(device)
        cand_text = candidate_prefix + candidate
        cand_ids = tok(
            cand_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        prefix_len = prefix_ids.shape[1]
        cand_len = cand_ids.shape[1]
        if cand_len == 0:
            return ScoreRecord(sum_logprob=0.0, n_tokens=0)

        full_ids = torch.cat([prefix_ids, cand_ids], dim=1)
        logits = self.model(full_ids).logits  # [1, T, V]
        # logits[i] predicts token i+1, so the candidate tokens at positions
        # [prefix_len, ..., prefix_len + cand_len - 1] are scored by logits at
        # [prefix_len - 1, ..., prefix_len + cand_len - 2].
        slice_start = prefix_len - 1
        slice_end = prefix_len + cand_len - 1
        cand_logits = logits[0, slice_start:slice_end, :]
        log_probs = F.log_softmax(cand_logits.float(), dim=-1)
        token_lp = log_probs.gather(1, cand_ids[0].unsqueeze(-1)).squeeze(-1)
        return ScoreRecord(sum_logprob=token_lp.sum().item(), n_tokens=cand_len)


def load_model(
    name: str,
    hf_id: str,
    dtype: str = "bfloat16",
    device_map: str | dict = "auto",
    trust_remote_code: bool = False,
    token: str | None = None,
) -> LoadedModel:
    """Load a causal LM and its tokenizer from the HF hub.

    `token` is forwarded to `from_pretrained` so gated models (Llama,
    Mistral, Gemma) can be downloaded once their access request is
    approved on the Hub. Pass the result of `ptc.env.get_hf_token()`.
    """
    torch_dtype = _DTYPE_MAP[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=trust_remote_code, token=token
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    model.eval()
    device = next(model.parameters()).device
    return LoadedModel(
        name=name, hf_id=hf_id, model=model, tokenizer=tokenizer, device=device
    )
