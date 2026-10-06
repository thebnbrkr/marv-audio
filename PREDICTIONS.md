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

## E2: which writes relay the audio (registered 2026-10-06)

Setup as E1 (same 20 clips, same teacher-forced transcripts, same 1 s silence
baseline). Script: `scripts/audio_routes.py`. All numbers are TOTAL effects
from real forward passes with writes swapped between the real-audio run and
the silence run. Per token, in log-probability of the token:

- S = silence run. A = real run. T = A − S, the whole audio effect.
- The audio reaches the decoder only through cross-attention, so the silence
  run with every cross-attention write taken from the real run must equal A.
- C = that, but with every self-attention and MLP write held at its silence
  value: the audio can only act through the cross-attention writes
  themselves. **direct = C − S.**
- B = that, with only the MLP writes held at silence values.
  **via MLP = A − B**, **via self-attention = B − C**. The three add up to T
  exactly. The split depends on the order in which routes are closed, so the
  other order (self-attention first) is reported too.
- Per layer: the real run with one layer's cross-attention write taken from
  the silence run. **loss_L = A − that.**

**P1 (sanity, exact).** Real cross-attention writes patched into the silence
run reproduce the real run, and silence writes patched into the real run
reproduce the silence run: max |Δ log p| < 1e-4. *Refuted if* either fails,
and then nothing else is reported.

**P2 (MLPs relay more than the direct route).** Summed over tokens, the
via-MLP share of T exceeds the direct share, in both orders. Basis: E1's
direct split gave MLP writes 53% of the logit against cross-attention's 29%.
*Refuted if* direct ≥ via-MLP in either order.

**P3 (no single layer carries it).** For the median token, removing any one
layer's cross-attention costs less than half of T. *Refuted if* some layer's
median loss_L / T is at least 0.5.

### E2 outcomes (run 2026-10-06, CPU, `results/e2_audio_routes.json`)

571 tokens, 523 with T > 1 nat; the audio effect sums to 3009 nats.

**P1 — CONFIRMED.** Swapping cross-attention writes reproduces both runs
exactly (max |Δ log p| = 0.0).

**P2 — REFUTED.** The audio acts mostly through the cross-attention writes
themselves. Share of T, summed over tokens:

| route | MLP closed first | self-attention closed first |
|---|---|---|
| direct (cross-attention writes only) | 84% | 84% |
| relayed by later MLPs | 16% | 12% |
| relayed by later self-attention | 0.6% | 4% |

Median direct share per token (T > 1 nat): 96%.

**P3 — CONFIRMED, narrowly.** Median loss from removing one layer's
cross-attention, as a share of T: layer 0 1%, layer 1 6%, layer 2 2%,
**layer 3 40%**. No layer reaches half, but the last layer carries far more
than the rest combined, so "no single layer carries it" is true only in the
weak sense the prediction tested.

**This corrects the reading of E1's P3.** E1 found cross-attention's direct
share of a token's logit (measured against the mean logit) to be 30%, while
silence costs 4.6 nats, and I took the gap to mean the audio is relayed
indirectly. E2 shows it is not: 84% of the audio's effect is direct. The two
numbers measure different things. E1's share is of the whole logit, most of
which is the language prior that is there with or without audio; E2's is of
the change the audio makes. E1's prediction (direct share below 50%) still
stands as measured; the interpretation attached to it does not.

## Notes from an external review (2026-10-06)

**Pre-registration trail.** E1 and E2 were each registered and resolved in
the same commit, so git cannot show the predictions came first. From E3 on,
PREDICTIONS.md is committed before the experiment runs; the commit
timestamp is the evidence.

**E1 P4, stress-tested.** The 0.72 correlation could have reflected tokens
with large logits having large everything. Controlling for the logit size
(partial Spearman, ranks residualised on `actual`) gives 0.78; using the
cross-attention *share* instead of its contribution gives 0.75. Recomputed
from `results/e1_listen_vs_guess.json`. P4 holds.

**E1 P3, the inference does not follow (second note).** P3's registered
criteria compared a fraction (direct share of the logit, 30%) with an
absolute amount (4.6 nats), so they never tested "understates". On a common
scale the medians are similar: direct cross-attention contribution 5.7 logit
units, total silence effect 4.6 nats (not identical units, and
cross-attention writes something on silence too). E2's 84% direct share
already contradicted the inference. The matched-units test is the
difference of two exact splits (real clip minus silence, same tokens), which
adds up exactly; it is a candidate for E3.

## E3: the audio's effect in matched units (registered 2026-10-06, committed before the run)

Setup as E1 and E2 (same 20 clips, same teacher-forced transcripts, same 1 s
silence). Script: `scripts/matched_split.py`. For each token, `listening_split`
runs on the real clip and on silence, scoring the same token at the same
position. Each split adds up exactly to that run's logit − mean logit, so the
difference does too:

    Δactual = Δembed + Δself_attn + Δcross_attn + Δmlp + Δbias

These are DIRECT effects in logit units, the matched-units counterpart of E2's
causal routes. Caveat: the final LayerNorm's scale differs between the two
runs, so part of every Δ is rescaling rather than new content. Shares below
are sums over all tokens (Σ Δpart / Σ Δactual).

**P1 (exact).** Every split's check error is below 1e-4, and the Δ parts add
up to Δactual within 1e-3 logit units for every token. *Refuted if* either
fails, and then nothing else is reported.

**P2 (cross-attention carries most of it).** Δcross_attn's share of Σ Δactual
is above 50%. Basis: E2's 84% direct route. *Refuted if* ≤ 50%.

**P3 (MLPs a minority).** Δmlp's share is below 30%. Basis: E2's 12–16% MLP
relay. *Refuted if* ≥ 30%.

**P4 (logit units track log-probability).** Across tokens, Δactual and E1's
total effect (Δ log p) are rank-correlated with Spearman above 0.7.
*Refuted if* ≤ 0.7.

### E3 outcomes (run 2026-10-06 after commit 0a29c15, CPU, `results/e3_matched_split.json`)

571 tokens; MARV `0.2.0+d405551` (recorded in the results file). Σ Δactual =
5638 logit units (median per token 9.6).

**P1 — CONFIRMED.** Largest split check error 1.9e-05; largest gap between the
summed Δ parts and Δactual 3.2e-05.

**P2 — CONFIRMED.** Δcross_attn = **123%** of Σ Δactual.

**P3 — CONFIRMED as worded, but the wording missed the sign.** Δmlp = **−25%**.
The MLP writes push the scored token *less* with the real audio than on
silence. The prediction ("below 30%") assumed a small positive relay; a
negative share satisfies it without testing that. What it shows: on silence
the MLPs supply more of the push toward the (now unheard) token, and with
audio, cross-attention supplies more than all of it while the MLPs back off.
The LayerNorm-scale caveat applies: part of each Δ is rescaling.

**P4 — CONFIRMED.** Spearman 0.90 between Δactual and E1's Δ log p.

The bias row's Δ is exactly 0: the LayerNorm bias contributes the same
amount in both runs. Self-attention 1.4%, embedding 0.2%.

Together with E2: the audio's effect is carried by cross-attention, both
causally (E2, 84% through the direct route) and in matched direct units (E3,
123%), with the MLPs partly offsetting it rather than relaying it.
