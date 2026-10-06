# marv-audio

[MARV](https://github.com/thebnbrkr/marv) for speech models, starting with Whisper.
Importing `marv_audio` registers a Whisper
decoder adapter, and MARV's own tools then work on Whisper: `capture_writes`,
`decompose_logit`, `patch_sweep`, `trace_by_depth`, `suppress` / `ablate`.
Every decomposition keeps MARV's exactness check.

## What is different about Whisper

A Whisper decoder layer adds **three** writes to the residual stream, not two:

| part | what it reads |
|---|---|
| `self_attn` | the text written so far |
| `cross_attn` | the audio (the encoder's output) |
| `mlp` | the residual so far (`fc1` → GELU → `fc2`) |

The final norm is a LayerNorm, whose bias becomes a constant row of its own,
and `proj_out` shares its weights with the token embedding. Only the decoder
is covered. Encoder features need labels from the audio itself (phonemes,
silence, noise), not a logit lens.

## Quick start

```python
import marv, marv_audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").eval()

ids = marv_audio.prompt_ids(proc) + proc.tokenizer.encode(" Hello world", add_special_tokens=False)
inputs = marv_audio.whisper_inputs(proc, audio, ids)        # audio: float32 at 16 kHz

marv.decompose_logit(model, None, inputs, target=ids[-1]).show()   # rows add up exactly
splits = marv_audio.listening_split(model, inputs)                 # per token: embed / self_attn / cross_attn / mlp / bias
marv_audio.language_logits(model, proc, audio)                     # language ID, read after <|startoftranscript|> only
```

## First result (E1, `PREDICTIONS.md`)

On whisper-tiny over 20 LibriSpeech clips (571 tokens, every split exact to
1e-6):

- Silence instead of the audio lowers the median token's log-probability by
  4.6 nats. Function words and word endings barely need the audio (" of",
  " a", "ering": under 0.3 nats). Names and rare words need it most
  (" Fred": 15 nats).
- Word starts depend on the audio more than word continuations (6.0 vs 3.6
  nats median).
- Cross-attention's **direct** share of a token's logit is only about 30%, yet
  removing the audio costs 4.6 nats. So the direct split alone understates how
  much the model listens, and the silence comparison is needed alongside it.
  Which later writes relay the audio is the next experiment (E2).

```bash
python scripts/listen_vs_guess.py      # ~1 min on a laptop CPU; downloads ~160 MB
```

## Layout

```
marv_audio/
  whisper.py   WhisperDecoderAdapter (registered on import), whisper_inputs,
               prompt_ids, language_logits
  listen.py    listening_split: exact per-token direct split, one forward pass
scripts/listen_vs_guess.py   E1
tests/test_whisper_exact.py  tiny random Whisper, no network: every decomposition exact
```

## Test

```bash
pip install -e git+https://github.com/thebnbrkr/marv#egg=marv   # or pip install -e ../marv
python -m pytest -q
```

## License

MIT. See `LICENSE`.
