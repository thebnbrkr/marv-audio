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

## Notebooks (Colab, free CPU)

| notebook | what |
|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thebnbrkr/marv-audio/blob/main/notebooks/01_tutorial_whisper.ipynb) `01_tutorial_whisper` | tutorial: transcribe a clip, check the writes add up, split each token into listening vs guessing, language ID, switch off neurons |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thebnbrkr/marv-audio/blob/main/notebooks/02_reproduce_experiments.ipynb) `02_reproduce_experiments` | rerun E1 and E2 and plot them |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thebnbrkr/marv-audio/blob/main/notebooks/tests_colab.ipynb) `tests_colab` | the test suite plus exactness checks on real whisper-tiny |

## Install

```bash
pip install "marv-audio[experiments] @ git+https://github.com/thebnbrkr/marv-audio"
```

This pulls MARV from GitHub. Don't `pip install marv`: that name on PyPI is an
unrelated robotics project.

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
marv_audio.language_probs(model, proc, audio)                      # language ID over all languages, read after <|startoftranscript|> only
```

## First results (E1, E2; see `PREDICTIONS.md`)

On whisper-tiny over 20 LibriSpeech clips (571 tokens, every split exact to
1e-6):

- Silence instead of the audio lowers the median token's log-probability by
  4.6 nats. Function words and word endings barely need the audio (" of",
  " a", "ering": under 0.3 nats). Names and rare words need it most
  (" Fred": 15 nats).
- Word starts depend on the audio more than word continuations (6.0 vs 3.6
  nats median).
- Most of each token's logit is language prior: cross-attention's direct share
  of the whole logit is about 30%. But of the *change* the audio makes, 84% acts
  directly through the cross-attention writes, and only 12–16% is relayed by
  later MLPs (E2). The last decoder layer's cross-attention matters most:
  removing it alone costs 40% of the audio effect for the median token.
- In matched units (E3: the exact split on the real clip minus the split on
  silence), cross-attention accounts for 123% of the change and the MLPs for
  −25%: without audio the MLPs guess harder, with audio they back off.

```bash
python scripts/listen_vs_guess.py      # E1, ~1 min on a laptop CPU; downloads ~160 MB
python scripts/audio_routes.py         # E2, same clips
python scripts/matched_split.py        # E3, same clips
```

## Layout

```
marv_audio/
  data.py      librispeech_clips: a small LibriSpeech sample (soundfile, any OS)
  whisper.py   WhisperDecoderAdapter (registered on import), whisper_inputs,
               prompt_ids, language_probs
  listen.py    listening_split: exact per-token direct split, one forward pass
scripts/listen_vs_guess.py   E1: listening vs guessing
scripts/audio_routes.py      E2: which writes relay the audio
scripts/matched_split.py     E3: the audio's effect split exactly in matched units
tests/test_whisper_exact.py  tiny random Whisper, no network: every decomposition exact
notebooks/                   Colab: tutorial, reproduction, tests
```

## Test

```bash
pip install -e ".[dev,experiments]"   # pulls MARV from GitHub
python -m pytest -q
```

## License

MIT. See `LICENSE`.
