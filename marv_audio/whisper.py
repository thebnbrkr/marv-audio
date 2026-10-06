"""Whisper decoder adapter for MARV, plus the audio-specific helpers.

A Whisper decoder layer adds THREE writes to the residual stream, not two:

    h = h + self_attn(self_attn_layer_norm(h))            # reads earlier text
    h = h + encoder_attn(encoder_attn_layer_norm(h), audio)  # reads the audio
    h = h + fc2(gelu(fc1(final_layer_norm(h))))            # the MLP

so   final = (token + position embedding) + sum of all three writes per layer
and  logits = proj_out(layer_norm(final)),

with `layer_norm` a LayerNorm (it has a bias) and `proj_out` tied to the token
embedding. MARV's trace tools handle both once this adapter is registered,
and every decomposition keeps its exactness check.

Only the decoder is covered. The encoder never produces vocabulary logits,
so its features need labels from the audio itself (phonemes, silence,
noise), not a logit lens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from marv.arch import ArchAdapter, register_adapter

PARTS = ("self_attn", "cross_attn", "mlp")


@register_adapter
class WhisperDecoderAdapter(ArchAdapter):
    """WhisperForConditionalGeneration: model.model.decoder.layers[i] with
    self_attn, encoder_attn and fc1/fc2; model.model.decoder.layer_norm;
    model.proj_out."""

    parts = PARTS

    @classmethod
    def matches(cls, model) -> str | None:
        cfg = getattr(model, "config", None)
        if getattr(cfg, "model_type", None) != "whisper":
            return "not a Whisper model"
        if not (hasattr(model, "proj_out") and hasattr(getattr(model, "model", None), "decoder")):
            return "Whisper without model.model.decoder / proj_out (load WhisperForConditionalGeneration)"
        if cfg.activation_function != "gelu":
            return f"activation_function is {cfg.activation_function!r}, not gelu"
        return None

    def layers(self, model) -> nn.ModuleList:
        return model.model.decoder.layers

    def component(self, model, layer: int, part: str) -> nn.Module:
        blk = model.model.decoder.layers[layer]
        if part == "self_attn":
            return blk.self_attn
        if part == "cross_attn":
            return blk.encoder_attn
        if part == "mlp":
            return blk.fc2  # fc2's output (bias included) is the MLP's whole write
        raise ValueError(f"part must be one of {self.parts}, got {part!r}")

    def ffn_in(self, model, layer: int) -> nn.Module:
        return model.model.decoder.layers[layer].fc1  # its input is final_layer_norm(h)

    def ffn_out(self, model, layer: int) -> nn.Linear:
        return model.model.decoder.layers[layer].fc2

    def final_norm(self, model) -> nn.Module:
        return model.model.decoder.layer_norm

    def unembed(self, model) -> torch.Tensor:
        return model.proj_out.weight.detach()

    def inputs(self, tokenizer, prompt, device: str) -> dict:
        if not isinstance(prompt, dict):
            raise TypeError(
                "Whisper needs audio: pass a dict such as whisper_inputs(processor, audio, decoder_ids), "
                "not a string"
            )
        return super().inputs(tokenizer, prompt, device)

    def embed(self, model) -> torch.Tensor:
        return model.model.decoder.embed_tokens.weight.detach()


def whisper_inputs(processor, audio, decoder_ids, sampling_rate: int = 16000) -> dict:
    """Model inputs for one clip: log-mel features (padded to 30 s, which
    Whisper requires) and the decoder token ids, as a batch of one."""
    feats = processor.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
    return {"input_features": feats, "decoder_input_ids": torch.as_tensor([list(decoder_ids)])}


def prompt_ids(processor, language: str = "en", task: str = "transcribe", timestamps: bool = False) -> list[int]:
    """The decoder prefix Whisper expects before any text:
    <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>."""
    tok = processor.tokenizer
    ids = [tok.convert_tokens_to_ids("<|startoftranscript|>"),
           tok.convert_tokens_to_ids(f"<|{language}|>"),
           tok.convert_tokens_to_ids(f"<|{task}|>")]
    if not timestamps:
        ids.append(tok.convert_tokens_to_ids("<|notimestamps|>"))
    return ids


def language_logits(model, processor, audio, languages=("en", "zh", "de", "es", "fr")) -> dict[str, float]:
    """Language identification as Whisper does it: the next-token logits after
    <|startoftranscript|> ALONE, read at the language tokens. Putting a
    language token in the input first would hand the model the answer."""
    tok = processor.tokenizer
    inputs = whisper_inputs(processor, audio, [tok.convert_tokens_to_ids("<|startoftranscript|>")])
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return {lang: float(logits[tok.convert_tokens_to_ids(f"<|{lang}|>")]) for lang in languages}
