# Predictions

Registered before the test runs. Outcomes are appended below each one; a
prediction is never edited after its test has run.

## E1: listening vs guessing in whisper-tiny (registered 2026-10-06)

Setup: `openai/whisper-tiny`, the first 20 clips of LibriSpeech
(`hf-internal-testing/librispeech_asr_dummy`, clean validation). Each clip is
transcribed greedily, then the model's own transcript is teacher-forced and
every predicted token is scored. Script: `scripts/listen_vs_guess.py`.

Two measures per token, kept apart:

- **direct**: the cross-attention writes' share of the token's logit (minus
  the mean logit), from `listening_split`. Exact, checked per token.
- **total**: log p(token | real audio) − log p(token | 1 s of silence), same
  decoder prefix. Includes every indirect path the audio takes.

**P1 (exactness).** Every token's split adds up to its real logit: maximum
check error below 1e-4. *Refuted if* any token exceeds it, and then no other
number from the run is reported.

**P2 (word pieces are guessed).** Tokens that continue a word (no leading
space) depend less on the audio than tokens that start a word: median total
effect lower for continuations. *Refuted if* the continuation median is at
or above the word-start median.

**P3 (direct is not total).** The direct cross-attention share understates
how much the model uses the audio, because audio read in early layers is
carried by later MLP and self-attention writes. Expect the median direct
share below 50% while silence lowers the median token log-probability by
more than 2 nats. *Refuted if* the direct share's median is at least 50%.

**P4 (the measures agree on order).** Across tokens, direct cross-attention
contribution and total audio effect are positively rank-correlated
(Spearman > 0.2). *Refuted if* ≤ 0.2.

### E1 outcomes (run 2026-10-06, CPU, `results/e1_listen_vs_guess.json`)

20 clips, 571 predicted tokens, mean WER 0.12 against the LibriSpeech
references (the model works; clip 4 is the noisiest at 0.21 over 93 tokens).

**P1 — CONFIRMED.** Maximum check error 1.2e-06 over 571 tokens.

**P2 — CONFIRMED.** Median total audio effect: word starts 6.04 nats
(n = 443), continuations 3.61 (n = 57). Unchanged without clip 4 (6.04 vs
3.61, n = 377 / 41). Continuations are few, so treat the size of the gap as
rough. Punctuation (1.39, n = 51) and end-of-text (1.16, n = 20) depend on the
audio least of all.

**P3 — CONFIRMED.** Median direct cross-attention share 30% of the token's
logit, while silence lowers the median token log-probability by 4.56 nats.
Summed over all tokens, the direct split of the logit is MLP 53%,
cross-attention 29%, LayerNorm bias 11%, self-attention 7%, embedding 0.3%.
Reading cross-attention's direct share alone would understate listening.
Where the rest of the audio's effect travels is NOT measured here: the MLP
writes carry the largest direct share, but how much of that is relayed audio
rather than text needs a path-patching run (E2: patch only the
cross-attention writes from the silence run).

**P4 — CONFIRMED.** Spearman 0.72 between direct cross-attention
contribution and total audio effect.

Least audio-dependent tokens: " of", " a", " to", " and", and word endings
("ering", "ish", "dle"), all under 0.3 nats. Most: names and rare words
(" Fred" 15.2, " lumin" 14.8, " decorative" 14.2, " disgrace" 14.1).

Caveats: one small model, 20 clips of one speaker set, the model's own
transcript (not the reference) teacher-forced, and "no audio" means one
second of silence padded to 30 s. A different no-audio baseline (noise,
another speaker's clip) could shift the total effect.
