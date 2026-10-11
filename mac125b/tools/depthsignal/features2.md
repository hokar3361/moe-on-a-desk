# Additional draft-time features for the per-draft stop rule (2026-10-11, offline)

Question: do features beyond S1_+mtp (MTP top-1 p, logit margin, weakest earlier draft, position, EMA, text position),
all computable at draft time in C++, raise held-out AUC and the replayed gain? Gate: replay gain >= +1.5 points over S1.

**Answer: no. Best combination +0.4 points (lambda\*), +0.9 points (online lambda). Does not pay; no new coefficients
are exported.** [計算, replay on held.log]

Data: `depthsignal-data/train.log` (48 prompts, 5502 rounds, 18451 samples), `held.log` (18 prompts, 1987 rounds, 6432
samples), depth fixed at 7. Fit on train only, scored on held. Replay = `analyze.py` simulator and cost model unchanged
(T(B) 1:17.9,3:25.8,6:34.3,8:41, draft 1.3 ms, host 1.0 ms, Dinkelbach lambda\*, option table, dmin 1).
Scripts: `features2.py` (features, cross-fitted token tables, ablation, replay via a `SignalStop` subclass),
`gbm_bound.py` (nonlinear upper bound). Reproduce: `python features2.py train.log held.log [--sets "...;..."]`.

## Text reconstruction

Per request the committed text is exact: round r contributes row-0 token (id_last) + accepted drafts; the next round's
row-0 token equals the previous round's target argmax at row L in 100 % of rounds and pos0 advances by L+1 in 100 %
(train and held). Generated length 187-399 tokens per request. **Prompt tokens are not in the log**, so the n-gram
features below only see the generated text; a C++ version could also index the prompt (not testable here).

## Feature groups tested (all causal: known before the verify of the current round)

| group | features (pre = before draft step k, post = after, d_k known) |
|---|---|
| inter | sum_{j<k} log p_j (cumulative MTP confidence), lg p_{k-1}, lg p_k x 1[k=1,2,3] (per-position slope) |
| nl | hinge basis on the MTP logit margin max(0, lm - t), t = 1, 2, 4, 8 |
| tail | MTP second prob p2 = p1 e^{-lm}, log of the top-10 mass outside the top two |
| ngram | longest suffix match (n <= 8) of text + d_1..d_{k-1} (pre); d_k == most recent continuation (hit), match but != d_k (miss), share of earlier continuations equal to d_k, log count, count of d_k in the text, d_k == d_{k-1} |
| tok | token tables learned on train (cross-fitted in 4 folds): shrunk mean residual y - q_S1 of the draft token d_k and of the context token before it |
| run | consecutive full-accept rounds, rounds since the last rejection, last round accepted/depth, EMA of that ratio (under the replayed policy's depth) |
| resid | per-request EMA of (y - q_S1) over tried positions (online calibration drift) |
| ver | S2 verifier features of the previous round (target margin, entropy, accept count, rejection) |

## Results (held-out, 18 prompts)

depth_ctl replay 83.0 tok/s, oracle 106.5 (+28.3 %).

| model | nf | AUC | k=1 | k=2 | k=3 | k>=4 | pre AUC | Brier | lambda\* tok/s | vs ctl | online lambda | prompts faster than S1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S1 (deployed)** | 9 | 0.888 | 0.888 | 0.888 | 0.878 | 0.898 | 0.671 | 0.1203 | 90.9 | +9.5 % | 89.2 | - |
| S1+inter | 14 | 0.890 | 0.888 | 0.889 | 0.879 | 0.898 | 0.672 | 0.1195 | 90.8 | +9.5 % | 89.3 | 6/18 |
| S1+nl | 13 | 0.891 | 0.887 | 0.891 | 0.880 | 0.901 | 0.671 | 0.1184 | 91.0 | +9.7 % | 89.5 | 10/18 |
| S1+tail | 11 | 0.891 | 0.887 | 0.892 | 0.880 | 0.902 | 0.671 | 0.1181 | 91.1 | +9.8 % | 89.7 | 12/18 |
| S1+ngram | 17 | 0.888 | 0.886 | 0.888 | 0.878 | 0.899 | 0.670 | 0.1202 | 90.8 | +9.5 % | 89.3 | 9/18 |
| S1+tok | 11 | 0.888 | 0.886 | 0.890 | 0.883 | 0.895 | 0.666 | 0.1199 | 91.1 | +9.8 % | 89.4 | 12/18 |
| S1+run | 13 | 0.888 | 0.889 | 0.887 | 0.879 | 0.899 | 0.688 | 0.1200 | 90.4 | +9.0 % | 89.1 | 5/18 |
| S1+resid | 10 | 0.888 | 0.888 | 0.888 | 0.879 | 0.898 | 0.671 | 0.1202 | 90.9 | +9.5 % | 89.2 | 7/18 |
| S1+ver | 16 | 0.889 | 0.890 | 0.889 | 0.877 | 0.899 | 0.690 | 0.1193 | 91.0 | +9.7 % | 89.7 | 8/18 |
| S1+inter+nl+tail | 20 | 0.892 | 0.888 | 0.892 | 0.881 | 0.902 | 0.672 | 0.1176 | 91.0 | +9.7 % | 89.7 | 11/18 |
| S1+inter+nl+ngram | 26 | 0.892 | 0.887 | 0.891 | 0.880 | 0.902 | 0.671 | 0.1179 | 91.1 | +9.8 % | 89.7 | 12/18 |
| all groups | 42 | 0.894 | 0.890 | 0.896 | 0.883 | 0.902 | 0.699 | 0.1161 | 91.2 | +9.9 % | 90.1 | 10/18 |
| GBM, S1 only (nonlinear) | 9 | 0.889 | 0.884 | 0.886 | 0.882 | 0.900 | 0.663 | - | 90.9 | +9.5 % | - | - |
| GBM, S1 + inter/ngram/tok/run/resid/nl | 31 | 0.891 | 0.886 | 0.892 | 0.887 | 0.896 | 0.678 | - | 90.9 | +9.5 % | - | - |

Differences of 0.1-0.3 tok/s and the "faster" counts near 9/18 are noise. No group reaches the +1.5-point gate; the
gradient-boosting bound on the same information shows the logistic form is not the limit either.

## Why (n-gram copy signal, conditional on the MTP p)

Acceptance by MTP p bin and n-gram state (held-out; train looks the same):

| MTP p | no match (n<2) | hit (d_k = earlier continuation) | miss |
|---|---:|---:|---:|
| < 0.5 | 0.22 (598) | 0.00 (8) | 0.15 (131) |
| 0.5-0.8 | 0.45 (1002) | 0.43 (35) | 0.44 (193) |
| 0.8-0.95 | 0.69 (756) | 0.65 (26) | 0.65 (127) |
| 0.95-0.999 | 0.89 (1200) | 0.88 (85) | 0.81 (180) |
| >= 0.999 | 0.99 (1471) | 0.99 (370) | 0.98 (250) |

Marginally, hits are accepted 0.90 vs 0.72 without a match, but hits already carry p ~ 0.96: given p the copy signal adds
nothing. The MTP head reads the target's last hidden state, which already encodes the repetition [解釈]. Indexing the prompt
in C++ would extend coverage, but the same "already in p" mechanism makes a large gain unlikely [仮説, untested here].
Run-length / drift / per-request residual features also add nothing (the per-position EMA and p already absorb them);
they raise the pre-step AUC (0.67 -> 0.69) but not the replay.

## Side finding: lambda handling matters more than any feature [計算]

Same S1 models, held-out replay: fixed lambda from the training Dinkelbach fixed point (0.092 tok/ms) 90.8 tok/s;
fixed lambda 0.08 / 0.09 / 0.10 / 0.11 -> 90.7 / 90.8 / 90.9 / 90.5 (flat); per-round EMA of the realised rate
(online, lam0 = 0.078 or 0.092) 89.2-89.3. The per-round online lambda costs ~1.6 points because the realised rate of
one round is noisy and rises right after easy rounds (which makes the rule stop earlier exactly when acceptance is
high). If the C++ rule uses the per-round online lambda, replacing it by a constant ~0.09 tok/ms (or a slow average
over many rounds / across requests) is worth about +1.6 points in replay — more than all features above combined.

## Conclusion

- Additional draft-time features (n-gram copy, token identity tables, run-length/drift, per-request residual,
  interactions, nonlinearity, MTP tail mass, previous-round verifier) raise held-out AUC 0.888 -> 0.894 at most and the
  replay +9.5 % -> +9.9 % (lambda\*). Below the +1.5-point gate: keep S1 as deployed; no C++ coefficients exported.
- The remaining gap to the oracle (+28 %) is not reachable with information available at draft time in these logs;
  closing it needs a better draft (higher acceptance), not a better stop decision.
