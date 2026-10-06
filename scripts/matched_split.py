"""E3 (PREDICTIONS.md): the audio's effect on each token, split exactly into
parts in matched units: (split on the real clip) − (split on silence).

    python scripts/matched_split.py         # same 20 clips and tokens as E1, CPU

Each split adds up exactly to its run's logit − mean logit, so their
difference adds up exactly to the change in that quantity. DIRECT effects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from listen_vs_guess import MODEL, REVISION, greedy, load_clips, spearman, token_logprobs  # noqa: E402
from marv.history import code_versions  # noqa: E402
from marv_audio import listening_split, prompt_ids, whisper_inputs  # noqa: E402

PARTS = ["embed", "self_attn", "cross_attn", "mlp", "bias"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--out", default="results/e3_matched_split.json")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.manual_seed(0)
    processor = WhisperProcessor.from_pretrained(MODEL, revision=REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, use_safetensors=True).eval()
    prefix = prompt_ids(processor)
    silence = np.zeros(16000, dtype=np.float32)

    rows, max_check, max_gap = [], 0.0, 0.0
    for k, (audio, _ref) in enumerate(load_clips(args.clips)):
        ids = greedy(model, processor, audio, prefix)
        positions = list(range(len(prefix) - 1, len(ids) - 1))
        tokens = [ids[j + 1] for j in positions]
        real, quiet = whisper_inputs(processor, audio, ids), whisper_inputs(processor, silence, ids)
        sr = listening_split(model, real, tokens=tokens, positions=positions)
        sq = listening_split(model, quiet, tokens=tokens, positions=positions)
        total = token_logprobs(model, real, positions, tokens) - token_logprobs(model, quiet, positions, tokens)
        for a, b, t in zip(sr, sq, total):
            delta = {p: a.by_part[p] - b.by_part[p] for p in PARTS}
            d_actual = a.actual - b.actual
            gap = abs(sum(delta.values()) - d_actual)
            max_check = max(max_check, a.check_error, b.check_error)
            max_gap = max(max_gap, gap)
            rows.append({"clip": k, "position": a.position, "token": a.token, "delta_actual": d_actual,
                         "total_effect_logp": float(t), **{f"delta_{p}": v for p, v in delta.items()}})
        print(f"clip {k:>2}  {len(sr)} tokens")

    print(f"\nP1: max split check error {max_check:.1e}, max |sum of deltas - delta actual| {max_gap:.1e}")
    if max_check >= 1e-4 or max_gap >= 1e-3:
        print("P1 REFUTED: the matched split does not add up; not reporting anything else.")
        sys.exit(1)

    d_actual = np.array([r["delta_actual"] for r in rows])
    summary = {
        "model": MODEL, "revision": REVISION, "code": code_versions(), "tokens": len(rows),
        "max_split_check_error": max_check, "max_delta_sum_gap": max_gap,
        "sum_delta_actual": float(d_actual.sum()),
        "share_of_delta": {p: float(sum(r[f"delta_{p}"] for r in rows) / d_actual.sum()) for p in PARTS},
        "median_delta_actual": float(np.median(d_actual)),
        "spearman_delta_actual_vs_logp": spearman(d_actual, np.array([r["total_effect_logp"] for r in rows])),
    }
    print(json.dumps(summary, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "tokens": rows}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
