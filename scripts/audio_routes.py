"""E2 (PREDICTIONS.md): which writes relay the audio to the output.

    python scripts/audio_routes.py          # same 20 clips and tokens as E1, CPU

Every number is a TOTAL effect from a real forward pass in which some writes
are swapped between the real-audio run and a silence run. Per-token ratios
use only tokens whose whole audio effect T exceeds 1 nat (fixed before the
run): for tokens the audio barely moves, a share of T is noise over noise.
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

from listen_vs_guess import MODEL, REVISION, greedy, load_clips  # noqa: E402
from marv.trace import _first, all_components, component, replace_outputs  # noqa: E402
from marv_audio import prompt_ids, whisper_inputs  # noqa: E402


@torch.no_grad()
def run(model, inputs, patches=None, capture=False):
    """Log-probs (seq, vocab), with `patches` {component: full write} swapped
    in; with `capture`, also every component's full write."""
    cache, hooks = {}, []
    if capture:
        for c in all_components(model):
            hooks.append(component(model, *c).register_forward_hook(
                lambda _m, _i, o, c=c: cache.__setitem__(c, _first(o).detach().clone())))
    try:
        with replace_outputs(model, {c: (lambda v: (lambda x: v))(v) for c, v in (patches or {}).items()}):
            lp = torch.log_softmax(model(**inputs).logits[0].float(), -1)
    finally:
        for h in hooks:
            h.remove()
    return (lp, cache) if capture else lp


def pick(lp, positions, tokens):
    return np.array([float(lp[j, t]) for j, t in zip(positions, tokens)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--out", default="results/e2_audio_routes.json")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.manual_seed(0)
    processor = WhisperProcessor.from_pretrained(MODEL, revision=REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, use_safetensors=True).eval()
    prefix = prompt_ids(processor)
    silence = np.zeros(16000, dtype=np.float32)
    comps = all_components(model)
    of = {p: [c for c in comps if c[1] == p] for p in ("self_attn", "cross_attn", "mlp")}
    n_layers = len(of["cross_attn"])

    rows, sanity = [], []
    for k, (audio, _ref) in enumerate(load_clips(args.clips)):
        ids = greedy(model, processor, audio, prefix)
        positions = list(range(len(prefix) - 1, len(ids) - 1))
        tokens = [ids[j + 1] for j in positions]
        real, quiet = whisper_inputs(processor, audio, ids), whisper_inputs(processor, silence, ids)

        lp_a, ca = run(model, real, capture=True)
        lp_s, cs = run(model, quiet, capture=True)
        A, S = pick(lp_a, positions, tokens), pick(lp_s, positions, tokens)

        def swap(inputs, real_parts=(), quiet_parts=(), extra=None):
            p = {c: ca[c] for part in real_parts for c in of[part]}
            p.update({c: cs[c] for part in quiet_parts for c in of[part]})
            p.update(extra or {})
            return pick(run(model, inputs, p), positions, tokens)

        sanity.append(float(np.abs(swap(quiet, real_parts=["cross_attn"]) - A).max()))
        sanity.append(float(np.abs(swap(real, quiet_parts=["cross_attn"]) - S).max()))

        B = swap(quiet, real_parts=["cross_attn"], quiet_parts=["mlp"])  # MLP route closed
        B2 = swap(quiet, real_parts=["cross_attn"], quiet_parts=["self_attn"])  # self-attn route closed
        C = swap(quiet, real_parts=["cross_attn"], quiet_parts=["mlp", "self_attn"])  # both closed
        loss = [A - swap(real, extra={c: cs[c]}) for c in of["cross_attn"]]

        for i in range(len(tokens)):
            rows.append({
                "clip": k, "position": positions[i], "token": tokens[i], "T": A[i] - S[i],
                "direct": C[i] - S[i],
                "mlp_first": {"via_mlp": A[i] - B[i], "via_self": B[i] - C[i]},
                "self_first": {"via_self": A[i] - B2[i], "via_mlp": B2[i] - C[i]},
                "loss_by_layer": [float(lo[i]) for lo in loss],
            })
        print(f"clip {k:>2}  {len(tokens)} tokens  sanity {max(sanity[-2:]):.1e}")

    worst = max(sanity)
    print(f"\nP1 sanity: max |delta log p| {worst:.1e}")
    if worst >= 1e-4:
        print("P1 REFUTED: swapping writes does not reproduce the runs; not reporting anything else.")
        sys.exit(1)

    T = np.array([r["T"] for r in rows])
    big = T > 1.0

    def share(get):
        return float(np.sum([get(r) for r in rows]) / T.sum())

    summary = {
        "model": MODEL, "revision": REVISION, "tokens": len(rows), "tokens_T_over_1nat": int(big.sum()),
        "sanity_max_abs": worst,
        "total_audio_effect_sum_nats": float(T.sum()),
        "share_of_T_summed": {
            "direct": share(lambda r: r["direct"]),
            "mlp_first": {"via_mlp": share(lambda r: r["mlp_first"]["via_mlp"]),
                          "via_self": share(lambda r: r["mlp_first"]["via_self"])},
            "self_first": {"via_mlp": share(lambda r: r["self_first"]["via_mlp"]),
                           "via_self": share(lambda r: r["self_first"]["via_self"])},
        },
        "median_loss_share_by_layer": [
            float(np.median([r["loss_by_layer"][L] / r["T"] for r, b in zip(rows, big) if b]))
            for L in range(n_layers)
        ],
        "median_direct_share": float(np.median([r["direct"] / r["T"] for r, b in zip(rows, big) if b])),
    }
    print(json.dumps(summary, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "tokens": rows}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
