# Per-round draft depth from the verifier's own signal (D1 revisited)

Status: offline tooling ready, nothing measured on the Mac yet. Labels: [実測] measured, [計算] computed, [仮説] hypothesis.

Files here:
- `sigpatch.py` — opt-in per-round log (`LLAMA_SPEC_SIGNAL_LOG=1|2`), anchors + replace on `common/speculative.cpp`.
- `analyze.py` — parse logs, fit calibrated acceptance models, AUC, replay policies (fixed, current `depth_ctl`,
  EV planner 0023, the new stop rule, oracle) with the measured T(B) table; `selftest` runs end to end on synthetic logs.

## 1. Why per-round depth, and how much is on the table

- Round = verify(B = d+1) + d draft steps + host. [実測] T(B) ≈ 17.9 / 25.8 / 34.3 / 41 ms at B = 1 / 3 / 6 / 8,
  draft step ≈ 1.3 ms, host ≈ 1 ms. Marginal verify cost of draft k: ~3.95 ms (k = 1, 2), ~2.8 ms (k = 3..5), ~3.35 ms (k = 6, 7).
- [実測] acceptance: prose ~0.6-0.72 at position 1, ~0.45-0.55 deeper; code ~0.9-0.99.
- The current controller picks one depth from smoothed per-position acceptance; within a request all rounds look alike to it.
  [計算, teammate] an oracle that knows each round's outcome gains ~19 % on prose. [計算, synthetic, this tool] +26 % prose /
  +28 % code with min depth 1, +34 % prose if the oracle may also skip drafting (depth 0) in rounds whose first draft fails.
- The question is purely: how much of that gap can a causal signal recover? `analyze.py selftest --sweep` maps
  signal quality to recovered gap on synthetic data calibrated to the acceptance and round time above
  (its replay of the current controller gives 66.5 tok/s on synthetic prose vs 65.9 [実測] on the fixed metric):

  | S2 AUC (draft k accepted) | prose vs depth_ctl | share of oracle gap |
  |---|---:|---:|
  | 0.59 (no signal beyond position) | +2 % | 8 % |
  | 0.66 | +3.6 % | 14 % |
  | 0.76 | +5 to +6.6 % | 19-26 % |
  | 0.85 | +9 % | 36 % |

  [計算] Applied to the teammate's 19 % oracle: AUC 0.65 → ~+2-3 %, 0.75 → ~+4-5 %, 0.85 → ~+7 %. The +2 % at "no signal"
  is the better objective (§3) alone, not the signal.

## 2. Features available when the depth of round r+1 is decided

Decision points: before each MTP draft step k (should step k run?) and right after it (should draft k enter the verify batch?).

| Feature | Available | Cost | Source |
|---|---|---|---|
| position one-hot k = 1, 2, 3, ≥4 | always | 0 | — |
| EMA of acceptance at position k (what `depth_ctl` uses) | always | 0 | `dctl.a[k]` |
| MTP top-1 prob p_k of the draft just produced (softmax over the sampler's top-10; renormalised, so over-confident, calibration fixes that) | after step k | 0 | `cur_p->data[0].p` |
| MTP logit margin top1 − top2 (invariant to the top-10 renormalisation) | after step k | 0 | `cur_p->data[0/1].logit` |
| weakest earlier draft min_{j<k} logit(p_j) | after step k−1 | 0 | kept in the loop |
| target logit margin at row n_acc of the previous verify (the row whose argmax is this round's `id_last`) | at round start | one max/second-max pass over 248k logits ≈ 0.1 ms [仮説], 1 row | `llama_get_logits_ith(ctx_tgt, i_batch_beg + n_acc)` in `accept()` |
| mean target margin over rows 0..n_acc of the previous verify | at round start | n_acc+1 rows | same |
| target top-1 prob / entropy at that row | at round start | + one thresholded exp pass | same (level 2 of the log) |
| previous round: accepted count, "had a rejection" | at round start | 0 | `accept()` |
| first round of the request | at round start | 0 | `begin()` |
| position in the text log1p(pos0 − pos0_first) | at round start | 0 | `dp.pos0` |

Not used (and why): the target rows of the *current* round are only known after the verify, i.e. after the decision;
target rows beyond the first rejection were computed on a wrong prefix.

Feature sets compared by `analyze.py` (AUC on held-out prompts): S0 position + EMA (= today's controller), S1 + MTP
probabilities (= the EV planner's information), S3 + verifier only, S2 all. Exact order/transform: `featurize()`.

## 3. Stop rule

Objective: maximise long-run tokens per ms. By the renewal-reward argument (Dinkelbach), the optimal per-round policy
maximises `E[tokens] − λ·E[ms]` with λ = the optimal long-run rate, *not* the per-round ratio `E/T`.
Let q_k = P(draft k accepted | drafts 1..k−1 accepted, features), P_k = q_1···q_k (probability that draft k is accepted).

1. Before draft step k (p_k unknown): `q_pre = σ(w_pre·x_pre)`. Running the step buys the information p_k, so its value is
   the option value `E_{q_post | q_pre}[max(0, A_k(q_post))] − λ·c_draft`, where the spread of q_post given q_pre is
   tabulated from training data (`OptionTable`, 10 bins × 8 quantiles). Stop if ≤ 0.
2. After draft step k: `q_post = σ(w_post·x_post)`. Keep draft k if `A_k(q_post) > 0`, else stop (that step was wasted).
3. Look-ahead A_k(q) = max over m ≥ 1 of Σ_{j=k}^{k+m−1} [P_{k−1}·q·q̄_{k+1}···q̄_j − λ·(ΔT_v(j) + c_draft·[j > k])],
   with q̄_j the marginal conditional acceptance at position j. Needed because ΔT_v is not monotone (3.95, 3.95, 2.8, ...):
   a myopic rule stops at k = 2 where k = 3 would still pay.
4. λ: online EMA of realised tokens/ms per round, initialised to the controller's current estimate; offline the
   Dinkelbach fixed point. ΔT_v(j) from the controller's measured round times (`dctl.T`), so the cost side is the one
   already maintained and corrected (M15).
5. Bounds: depth ≤ cap (7), depth ≥ `LLAMA_SPEC_DEPTH_MIN` (1; depth 0 is a separate experiment, it needs a decision
   before any draft and only adds ~0.2 % in the synthetic replay).
6. Calibration: logistic regression with per-position intercepts, fitted offline on training prompts (never on the
   fixed-metric prompts or `heldout.json`), exported by `analyze.py fit --export` (μ, σ, w, option table, q̄). An online
   per-request bias term (one scalar, SGD on the log-loss of tried positions) can follow if held-out calibration drifts.

## 4. Plug-in to the existing draft loop (after the go decision; not implemented)

The MTP head already produces drafts one by one in `draft()`; each step decodes one token, so stopping is free.
- `accept()`: after `dctl.last_acc = n_accepted`, read `llama_get_logits_ith(ctx_tgt, i_batch_beg[seq] + n_accepted)`
  (still valid: no target decode happens in between; with backend sampling it is null → features default, flag set),
  compute the margin; store prev-round features per seq.
- `draft()`, per-step loop, right after `cur_p` is obtained and before `result.push_back(id)`: compute x_post, q_post,
  `A_k`; if ≤ 0 → `drafting[seq] = false; n_drafting--; continue` (same shape as the existing `p_min` stop). After the push,
  before queuing step k+1: compute q_pre for k+1 and the option value; if ≤ 0 → stop as the depth-cap check does.
- `depth_ctl` keeps measuring T(d) and supplies the cap and ΔT_v; `mtp_depth` becomes the cap only.
- Width changes every round are now cheap: the graph cache (M6, patch 0034) keeps one graph per ubatch shape.

## 5. What is different from the failed EV planner (patch 0023, `LLAMA_SPEC_EV=1`)

Read from the patch (no measured numbers are recorded in DESIGN.md beyond "slower as calibrated" / 7.1 "failed"):

| | EV planner 0023 | this design |
|---|---|---|
| Infrastructure | written before the graph cache (0034): a per-round width change rebuilt the 7k-node verify graph, ~3 ms/round CPU (DESIGN 9.4: the depth controller lost 61.3 → 53.8 tok/s on prose for the same reason) | graph cache in place; variable width costs 0.40 ms CPU [実測 9.4] |
| Cost model | hard-coded pre-M10 table 21 / 28.8 / 31.6 / 34.3 ... (B=3 then measured 33.9, not 31.6) + 1.5 ms/draft | the controller's measured, monotone T(d) (M15) |
| Objective | maximise this round's ratio E/T, myopic one-step; includes draft k if P_k/ΔT > E/T (E/T ≈ 1/22.5 ≈ 0.044 tok/ms at k = 1, well below the long-run rate ≈ 0.066) → too permissive early, and blind to the non-monotone ΔT_v | maximise E − λT with λ = long-run rate; multi-step look-ahead; explicit option value of one more draft step |
| Information | only the MTP top-1 p (top-10 renormalised) | + logit margin, weakest earlier draft, per-position EMA, the target's own margin/entropy at the row that starts the round, previous outcome, text position |
| Calibration | one global 20-bucket table on p, shared by all positions and by all requests (code and prose mixed), prior = p itself | logistic with per-position intercepts, fitted on separate training prompts, scored by AUC on held-out prompts before any code is written |
| Draft step | always run, then decide inclusion (always pays the step) | pre-decision can skip the step |
| Go/no-go | none (implemented directly) | AUC gate and replay gain gate (§6) on held-out prompts |

The replay includes a port of 0023: on synthetic prose it gives +2-4 % over `depth_ctl` vs +5-7 % for the stop rule at
the same signal [計算]. So 0023 was probably not wrong in kind; it most likely lost to the per-round graph rebuild and the
stale cost table [仮説]. A cheap first check needs no new code: `LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_COST=<current T(B)+host>
LLAMA_SPEC_EV_DRAFT_MS=1.3` on the current build, with the graph cache.

## 6. Protocol and gates

1. Collect logs at a fixed depth 7 (no depth controller, `LLAMA_SPEC_MTP_DEPTH=7`, no `--draft-p-min`): every round
   records 7 drafts, so the accepted count at any depth d ≤ 7 is exactly min(L, d) (greedy drafts are deterministic).
   Training prompts: `analyze.py prompts gendata2.py` (fixed-metric topics are excluded by gendata2's filter).
   Test prompts: `tools/mtp/heldout.json` (18, never used for training). Fixed-metric code/prose: evaluation only.
2. `analyze.py fit --train train.log --test held.log`. Gate 1: S2 AUC at k = 1 and overall ≥ 0.75 (DESIGN 9.15 #5);
   S2 − S1 tells whether the verifier adds anything. Gate 2: simulated stop rule (online λ) ≥ +4 % over the `depth_ctl`
   replay on held-out prose, without loss on code.
3. Only then implement §4, opt-in; check `verify.sh` md5 and the fixed-metric texts against -11d; fixed metric twice;
   held-out 18 prompts as the generality reference (rule 11).
4. Compare the replay with a log collected under the new rule (same patch, level 1) to measure the replay bias.

## 7. Risks

- The signal may be weak: MTP and target share the hidden state, so the target margin may add little over p_k
  (S2 ≈ S1). Then the gain is the objective fix only (+2 % synthetic) — below the noise of one fixed-metric run.
- Replay bias: rounds are replayed on the partition of the depth-7 run; under a shallow policy round starts fall at
  other positions (e.g. less often right after a rejection). The oracle shares the bias; step 6.4 measures it.
- Output identity: per-round width variation is already exercised by `depth_ctl`, but every new width pattern must still
  pass the text comparison (the q4_K MMA withdrawal showed B ≥ 2 numerics can change prose).
- Host cost: one 248k-logit pass per round (~0.1 ms [仮説]) and the logistic (negligible). The log itself (level 2:
  one exp pass per verify row) perturbs timing; never benchmark with it on.
- Logged p is the top-10-renormalised probability; if the sampler chain changes, retrain.
- Overfitting to the prompt family: fit on gendata2 prompts, gate on heldout.json, keep the fixed-metric prompts out.

## 8. Self-test (synthetic, [計算])

`python analyze.py selftest` (≈7 s): writes synthetic logs in the exact log format (latent AR(1) difficulty drives
acceptance; p_k and the target margin are noisy views), parses them, checks that the accept count re-derived from the
verify rows matches `nacc`, fits, reports AUC and the replay table, and fails if a causal policy beats the oracle.
At AUC ≈ 0.76: prose depth_ctl 66.5, EV-0023 68.9, stop rule 70.5-70.8, oracle 83.5 tok/s; code 95.1 / 100.2 / 101.8 / 122.3.
