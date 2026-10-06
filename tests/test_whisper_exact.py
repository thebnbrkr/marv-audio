"""Exactness tests on a tiny random Whisper: no network, no checkpoint.
The model can't transcribe anything. The point is that every decomposition
adds up to the real output, and every hook lands on the right module."""
from __future__ import annotations

import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

import marv_audio  # noqa: F401  (registers the adapter)
from marv.arch import detect_adapter
from marv.edit import ablate, restore, suppress
from marv.trace import all_components, capture_writes, decompose_logit, replace_outputs, trace_by_depth
from marv_audio import PARTS, WhisperDecoderAdapter, listening_split

N_LAYERS = 2


def tiny_whisper():
    cfg = WhisperConfig(
        d_model=32, encoder_layers=2, decoder_layers=N_LAYERS, encoder_attention_heads=4,
        decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64, vocab_size=128,
        num_mel_bins=16, max_source_positions=50, max_target_positions=32,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=3,
    )
    torch.manual_seed(0)
    m = WhisperForConditionalGeneration(cfg).eval()
    with torch.no_grad():  # a fresh LayerNorm is (1, 0); randomise so the norm maths is exercised
        for mod in m.modules():
            if isinstance(mod, torch.nn.LayerNorm):
                mod.weight.normal_(1.0, 0.3)
                mod.bias.normal_(0.0, 0.3)
    return m


def inputs(seed=0, ids=(3, 5, 7, 9, 11)):
    # the encoder requires exactly 2 * max_source_positions mel frames
    g = torch.Generator().manual_seed(seed)
    return {"input_features": torch.randn(1, 16, 100, generator=g),
            "decoder_input_ids": torch.tensor([list(ids)])}


def test_adapter_is_detected():
    assert isinstance(detect_adapter(tiny_whisper()), WhisperDecoderAdapter)


def test_three_writes_per_layer_rebuild_the_final_residual():
    m = tiny_whisper()
    w = capture_writes(m, None, inputs(), positions=[0, 2, 4])
    assert len(w.parts) == len(PARTS) * N_LAYERS
    assert w.reconstruction_error() < 1e-5


def test_missing_cross_attention_breaks_the_reconstruction():
    # the check must be able to fail: drop the audio writes and it does
    m = tiny_whisper()
    w = capture_writes(m, None, inputs())
    w.parts = {c: v for c, v in w.parts.items() if c[1] != "cross_attn"}
    assert w.reconstruction_error() > 1e-2


def test_logit_decomposition_through_layernorm_is_exact():
    m = tiny_whisper()
    for target, baseline in ((5, None), (7, 9), (100, 3)):
        d = decompose_logit(m, None, inputs(), target, baseline)
        assert d.check_error < 1e-4, d.check_error
        assert set(d.by_part()) == {"embed", "self_attn", "cross_attn", "mlp", "bias"}


def test_listening_split_is_exact_at_every_position():
    m = tiny_whisper()
    splits = listening_split(m, inputs())
    assert [s.position for s in splits] == [0, 1, 2, 3]
    assert [s.token for s in splits] == [5, 7, 9, 11]
    for s in splits:
        assert s.check_error < 1e-4, s.check_error


def test_suppress_and_ablate_give_identical_outputs():
    m = tiny_whisper()
    feats = [(1, 7), (0, 3)]
    with torch.no_grad():
        base = m(**inputs()).logits
        with suppress(m, feats):
            sup = m(**inputs()).logits
        saved = ablate(m, feats)
        abl = m(**inputs()).logits
        restore(m, saved)
        again = m(**inputs()).logits
    assert (sup - base).abs().max() > 0
    torch.testing.assert_close(sup, abl)
    torch.testing.assert_close(again, base)


def test_patching_every_write_from_other_audio_reproduces_it():
    m = tiny_whisper()
    src, tgt = inputs(seed=1), inputs(seed=2)
    cache = {}
    hooks = []
    from marv.trace import _first, component

    for c in all_components(m):
        hooks.append(component(m, *c).register_forward_hook(
            lambda _m, _i, o, c=c: cache.__setitem__(c, _first(o).detach().clone())))
    with torch.no_grad():
        want = m(**src).logits
    for h in hooks:
        h.remove()
    with torch.no_grad(), replace_outputs(m, {c: (lambda v: (lambda x: v))(cache[c]) for c in cache}):
        got = m(**tgt).logits
    torch.testing.assert_close(got, want)


def test_trace_by_depth_starts_at_exactly_one():
    m = tiny_whisper()
    src, tgt = inputs(ids=(3, 5, 7, 9)), inputs(ids=(3, 6, 7, 9))

    def metric(logits):
        return float(logits[0, -1, 20] - logits[0, -1, 21])

    rows = trace_by_depth(m, None, src, tgt, position=1, metric=metric)
    assert rows[0]["depth"] == -1
    assert abs(rows[0]["fraction"] - 1.0) < 1e-4


def test_a_string_prompt_is_refused():
    with pytest.raises(TypeError, match="audio"):
        capture_writes(tiny_whisper(), None, "hello")


def test_ffn_input_and_activations_are_read_where_the_mlp_sees_them():
    from marv.context import feature_activations_at_layers, hidden_states_at_layers

    m = tiny_whisper()
    seen = {}
    h = m.model.decoder.layers[1].final_layer_norm.register_forward_hook(
        lambda _m, _i, o: seen.__setitem__("x", o[0, -1].detach()))
    with torch.no_grad():
        m(**inputs())
    h.remove()
    x = torch.as_tensor(hidden_states_at_layers(m, None, inputs(), [1])[1])
    torch.testing.assert_close(x, seen["x"])
    a = feature_activations_at_layers(m, None, inputs(), [1])[1]
    blk = m.model.decoder.layers[1]
    torch.testing.assert_close(torch.as_tensor(a), blk.activation_fn(blk.fc1(x)).detach())
