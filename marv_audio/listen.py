"""Listening vs guessing: how much of each predicted token's logit came
straight from the audio.

Every decoder token's logit is a sum of direct pushes: the embedding, each
layer's self-attention (earlier text), cross-attention (the audio) and MLP
writes, and the final LayerNorm's bias. `listening_split` computes that sum
exactly for every position in one forward pass, with the check that the
parts add up to the real logit.

This is a DIRECT effect. Audio that enters through cross-attention in layer
1 and is then carried by a later MLP is credited to the MLP here. For the
total effect of the audio, compare against a run on silence (see
scripts/listen_vs_guess.py).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from marv.trace import _adapter, _frozen_norm, capture_writes


@dataclass
class TokenSplit:
    position: int  # decoder position whose logits are read
    token: int  # the token scored there (by default the true next token)
    actual: float  # logit[token] - mean logit, read from the real logits
    by_part: dict[str, float]  # direct share of `actual`: embed / self_attn / cross_attn / mlp / bias
    by_layer_part: dict[tuple[int, str], float]

    @property
    def total(self) -> float:
        return sum(self.by_part.values())

    @property
    def check_error(self) -> float:
        """|sum of parts - actual|, relative. ~0; if not, don't trust the parts."""
        return abs(self.total - self.actual) / max(abs(self.actual), 1e-9)


@torch.no_grad()
def listening_split(model, inputs: dict, tokens: list[int] | None = None,
                    positions: list[int] | None = None, device: str = "cpu") -> list[TokenSplit]:
    """Exact direct split of logit[token] - mean logit at each decoder position.

    `inputs` holds `input_features` and `decoder_input_ids` (see
    whisper_inputs). By default every position j is scored on the token at
    j + 1 in `decoder_input_ids` (teacher forcing on a reference transcript);
    pass `tokens` (one per position) to score something else.
    """
    ids = inputs["decoder_input_ids"][0].tolist()
    positions = list(range(len(ids) - 1)) if positions is None else list(positions)
    if tokens is None:
        tokens = [ids[j + 1] for j in positions]
    if len(tokens) != len(positions):
        raise ValueError("need one token per position")

    ad = _adapter(model)
    w = capture_writes(model, None, inputs, positions=positions, device=device)
    W = ad.unembed(model).float().cpu()
    norm = ad.final_norm(model)

    out = []
    for i, (j, t) in enumerate(zip(positions, tokens)):
        direction = W[t] - W.mean(0)
        f, const = _frozen_norm(norm, w.final[i])
        by_part: dict[str, float] = defaultdict(float)
        by_layer_part = {}
        by_part["embed"] = float(f(w.embed[i]) @ direction)
        for (layer, part), v in w.parts.items():
            x = float(f(v[i]) @ direction)
            by_layer_part[(layer, part)] = x
            by_part[part] += x
        by_part["bias"] = float(const @ direction)
        actual = float(w.logits[i, t] - w.logits[i].mean())
        out.append(TokenSplit(j, int(t), actual, dict(by_part), by_layer_part))
    return out
