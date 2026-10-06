# AGENTS.md

Guidance for AI coding agents (and humans) working in this repo.

## What this is

MARV for speech models (package `marv_audio`). MARV
(https://github.com/thebnbrkr/marv) is the engine; this package adds an
architecture adapter for the Whisper decoder plus audio-specific analyses.
Keep generic machinery in MARV and only Whisper/audio specifics here. If a
tool would work on any model, it belongs in MARV.

## The model facts everything depends on

Checked against transformers 4.57 `models/whisper/modeling_whisper.py`.

- Load `WhisperForConditionalGeneration`. Paths: `model.model.decoder.layers`,
  `model.model.decoder.layer_norm`, `model.proj_out`. There is no
  `model.decoder`.
- Decoder layer writes, in order: `self_attn`, `encoder_attn` (cross-attention,
  reads the audio), `fc2` (the MLP: `fc2(gelu(fc1(final_layer_norm(h))))`).
  Each output, biases included, is that part's whole residual write.
- Residual entering layer 0 = token embedding + positional embedding
  (`scale_embedding` is False for released checkpoints). final = that + all
  writes; logits = `proj_out(layer_norm(final))`, `proj_out` tied to
  `embed_tokens`, no bias.
- The final LayerNorm is exact under frozen statistics only with
  `sqrt(var(unbiased=False) + eps)`. `std + eps` is off by ~3e-3 relative.
- The encoder requires exactly `2 * max_source_positions` mel frames (3000
  for real checkpoints, 30 s). Every clip is padded to that, and each encoder
  position covers 20 ms.
- Language ID is read after `<|startoftranscript|>` alone. Never put a
  language token in the input when measuring it.

## Invariants

1. Every decomposition ships its check (`Writes.reconstruction_error`,
   `Decomposition.check_error`, `TokenSplit.check_error`). Never report numbers
   from a run whose check failed.
2. Direct ≠ total. `listening_split` / `decompose_logit` are direct. Silence or
   patching runs are total. Label every number.
3. `suppress` and `ablate` on the same features give identical outputs. Don't
   report them as two results.
4. Predictions are pre-registered in `PREDICTIONS.md`. Append outcomes; never
   edit a prediction after its test has run.
5. Tests use a tiny random Whisper with randomised LayerNorms (a fresh
   LayerNorm is (1, 0) and hides norm bugs). New behaviour gets an exactness
   test there.

## Build / test

```bash
python -m pytest -q                       # tiny model, CPU, no network
python scripts/listen_vs_guess.py         # E1 on whisper-tiny, CPU
```
