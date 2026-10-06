"""E1 (PREDICTIONS.md): how much of each token whisper-tiny writes comes from
listening to the audio, and how much from guessing from the text so far.

    python scripts/listen_vs_guess.py            # 20 LibriSpeech clips, CPU, ~1 min

Downloads whisper-tiny (~150 MB) and a 9 MB LibriSpeech sample. Decodes FLAC
with soundfile (falls back to macOS `afconvert`).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marv_audio import librispeech_clips, listening_split, prompt_ids, whisper_inputs  # noqa: E402

MODEL = "openai/whisper-tiny"
REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"


def load_clips(n: int):
    return librispeech_clips(n)


@torch.no_grad()
def greedy(model, processor, audio, prefix, max_new=120):
    eot = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ids = list(prefix)
    feats = whisper_inputs(processor, audio, ids)["input_features"]
    for _ in range(max_new):
        nxt = int(model(input_features=feats, decoder_input_ids=torch.tensor([ids])).logits[0, -1].argmax())
        ids.append(nxt)
        if nxt == eot:
            break
    return ids


def words(text):
    return re.sub(r"[^A-Z' ]", " ", text.upper()).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


@torch.no_grad()
def token_logprobs(model, inputs, positions, tokens):
    lp = torch.log_softmax(model(**inputs).logits[0].float(), -1)
    return np.array([float(lp[j, t]) for j, t in zip(positions, tokens)])


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--out", default="results/e1_listen_vs_guess.json")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.manual_seed(0)
    processor = WhisperProcessor.from_pretrained(MODEL, revision=REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, use_safetensors=True).eval()
    tok = processor.tokenizer
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    prefix = prompt_ids(processor)
    silence = np.zeros(16000, dtype=np.float32)

    cache = Path(args.out).parent
    cache.mkdir(parents=True, exist_ok=True)
    rows, clip_rows = [], []
    for k, (audio, ref) in enumerate(load_clips(args.clips)):
        ids = greedy(model, processor, audio, prefix)
        hyp = tok.decode(ids[len(prefix):], skip_special_tokens=True)
        positions = list(range(len(prefix) - 1, len(ids) - 1))
        tokens = [ids[j + 1] for j in positions]

        real = whisper_inputs(processor, audio, ids)
        quiet = whisper_inputs(processor, silence, ids)
        splits = listening_split(model, real, tokens=tokens, positions=positions)
        total = token_logprobs(model, real, positions, tokens) - token_logprobs(model, quiet, positions, tokens)

        clip_rows.append({"clip": k, "wer": wer(ref, hyp), "ref": ref, "hyp": hyp})
        for s, eff in zip(splits, total):
            piece = tok.convert_ids_to_tokens(s.token)
            kind = ("end" if s.token == eot else "word_start" if piece.startswith("Ġ")
                    else "punct" if re.fullmatch(r"[^\w]+", piece) else "continuation")
            rows.append({"clip": k, "position": s.position, "token": piece, "kind": kind,
                         "actual": s.actual, "check_error": s.check_error, "total_effect": float(eff),
                         **{f"direct_{p}": v for p, v in s.by_part.items()}})
        print(f"clip {k:>2}  WER {clip_rows[-1]['wer']:.2f}  {len(splits)} tokens  {hyp[:60]!r}")

    max_err = max(r["check_error"] for r in rows)
    print(f"\nP1 exactness: max check error {max_err:.1e} over {len(rows)} tokens")
    if max_err >= 1e-4:
        print("P1 REFUTED: decomposition does not add up; not reporting anything else.")
        sys.exit(1)

    actual = np.array([r["actual"] for r in rows])
    cross = np.array([r["direct_cross_attn"] for r in rows])
    total = np.array([r["total_effect"] for r in rows])
    share = cross / np.where(np.abs(actual) > 1e-6, actual, np.nan)
    parts = ["embed", "self_attn", "cross_attn", "mlp", "bias"]

    summary = {
        "model": MODEL, "revision": REVISION, "clips": len(clip_rows), "tokens": len(rows),
        "mean_wer": float(np.mean([c["wer"] for c in clip_rows])),
        "max_check_error": max_err,
        "direct_share_of_logit": {p: float(np.sum([r[f"direct_{p}"] for r in rows]) / actual.sum()) for p in parts},
        "median_direct_cross_share": float(np.nanmedian(share)),
        "median_total_effect_nats": float(np.median(total)),
        "spearman_direct_cross_vs_total": spearman(cross, total),
        "by_kind": {},
    }
    for kind in ("word_start", "continuation", "punct", "end"):
        sel = [r for r in rows if r["kind"] == kind]
        if sel:
            summary["by_kind"][kind] = {
                "n": len(sel),
                "median_total_effect_nats": float(np.median([r["total_effect"] for r in sel])),
                "median_direct_cross_share": float(np.nanmedian(
                    [r["direct_cross_attn"] / r["actual"] for r in sel if abs(r["actual"]) > 1e-6])),
            }

    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "clips": clip_rows, "tokens": rows}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
