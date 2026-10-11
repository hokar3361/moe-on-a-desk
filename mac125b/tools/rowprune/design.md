# Lossless mid-pass row pruning of the speculative verify pass — design and go/no-go

Status: proposal + offline measurement kit, nothing measured on the Mac yet. Numbers marked [calc] come from the
cost model below (DESIGN.md 9.13-9.15); nothing here is [measured] for this idea.

## 0. Summary

- A verify pass of B = k+1 rows `[t0, d1..dk]` only needs row j if d1..dj are all accepted. If, at layer L, a probe
  predicts that the target will reject d_{j+1}, every row > j can stop paying for its MoE for layers > L. A wrong
  prediction never changes the output; it only shortens the round (the tokens after the cut are re-drafted).
- Only the MoE (row, expert) pairs are worth skipping: for B >= 3 about 94 % of the per-row verify cost is MoE
  (B=4→8: +11.5 ms, of which MoE +10.6 ms, DESIGN 9.13), ≈ 2.65 ms per row per pass, ≈ 55 µs per row per layer.
- Ceiling with a perfect probe [calc]: prose (depth 2) +6.6 % at il=15 (cut after 16 of 48 layers), +4.8 % at il=23,
  +3.2 % at il=31; code (depth 5) +5.4 / +4.0 / +2.6 %, +7.3 % at il=15 if the depth controller then goes to depth 7.
- The same perfect information used *before* the verify (do not draft the tokens that will be rejected) is worth
  +16 % (prose) / +12-20 % (code) [calc], because it removes whole rows and their draft steps. So the mid-pass probe only
  has value through information that the MTP's own confidence does not already carry.
- False drops are expensive: each 1 % of falsely cut accept decisions costs ≈ 0.7 % (prose, depth 2) and ≈ 2.2 % (code,
  depth 5) of the tokens [calc] — a false cut at an early row discards every later accepted token of the round.
- Go/no-go (section 6): GO only if a deployable probe at il ≤ 23 gives, on held-out prompts, ≥ +3 % on prose and ≥ +2 %
  on code in the simulator, AND still adds ≥ +2 / +1 points on top of pre-verify truncation by draft confidence. My prior
  is NO-GO (draft errors concentrate on tokens where the target itself is undecided at 1/3-1/2 depth), but the test is
  cheap: one dump run on the Mac (~1 h) plus probe.py on the 4090.

## 1. Exact semantics

### 1.1 Rows and labels
Verify batch rows j = 0..k: row 0 is the last sampled token t0 (not yet in the cache), row j ≥ 1 is draft d_j. With greedy
decoding the target accepts n = max m such that argmax(row j-1) = d_j for all j ≤ m, emits d_1..d_n plus
c = argmax(row n), and commits rows 0..n to the KV / recurrent state. Row j "decides" d_{j+1}:
y_j = [argmax(row j) ≠ d_{j+1}]. Row j is alive (its output matters) iff j ≤ n.

### 1.2 Only suffixes can be cut
Row j+1.. depend on row j through (a) causal attention in the full-attention layers (K/V of row j at every layer),
(b) the Gated-DeltaNet recurrent state and the conv state (row j+1 starts from the state after row j), (c) the QSA
indexer (pooled keys of a completed block include row j). Rows < j are not affected by anything in rows ≥ j: attention is
causal, the recurrences run forward, and the MoE / dense / HC-mixer arithmetic of a row reads only that row (given a fixed
batch shape, see 1.4). Hence the only consistent operation is a cut r ∈ [1, B]: rows 0..r-1 exact, rows ≥ r undefined
after layer L. Row 0 is never cut.

Decision rule on the GPU at layer L: r = 1 + min{ j ∈ [0, B-2] : score_j > τ }, else r = B. A flag on a row that is
already dead (j > n) is free; a flag on row n is a correct catch; a flag on j < n is a false drop.

### 1.3 What the round emits
- n < r: identical to the uncut round (rows 0..n are exact).
- n ≥ r: rows 0..r-1 are exact and argmax(row r-1) = d_r, so d_1..d_r are emitted (r tokens instead of n+1), the state is
  committed through row r-1, and d_r becomes the next round's first row (its own row was cut, so its K/V and recurrent
  update are recomputed next round).
- Host side this is exactly "the draft was d_1..d_{r-1}": call the existing acceptance with `spec_draft` truncated to r-1
  and `spec_i_batch` to r entries; the rollback count is computed from the original draft length; the KV entries and
  recurrent snapshots of rows ≥ r are discarded by the existing rollback (they are no different from rejected rows, the
  bounded snapshots of M4 cover every row of the batch). The MTP hand-off uses rows 0..r-1 only (merged catch-up M8 reads
  the h of accepted rows), so it stays consistent.
- The depth controller must see positions ≥ r as *censored* (not observed), not as rejected; otherwise it learns a lower
  acceptance and shortens the draft.

### 1.4 Exactness
Every emitted token is the argmax of a row whose computation is bit-identical to the uncut verify of the same width,
provided that (i) the cut is data (a mask), not shape: B, the graph, the kernel choices (they depend on ne11/ne21 and the
pair count) and the graph cache key stay the same; (ii) every MoE kernel computes each (row, expert) pair independently of
which other pairs are live (true for `kernel_mul_mv_id` with the ballot dedup — the owner threadgroup loops over the matching
pairs and calls the same per-pair routine — and must be checked for the q2_0 few-token kernel (patch 0032) and the opt-in
iq4_nl moe_mt kernel (0020)); (iii) cut rows stay finite (1.5). What does change is the round boundaries: after a cut,
d_r is recomputed as row 0 of the next batch at another width. That is the same class of effect the depth controller
already has (verify-width numerics, DESIGN 9.13 q4_K withdrawal) and is checked the same way: verify.sh md5 and identical
fixed-metric texts.

### 1.5 Cut rows must stay finite
Masked-out rows are still read with weight 0 by live rows in places such as the flash-attention V accumulation
(`o = o*ms + p*v` with p = 0) — 0·NaN = NaN would poison live rows. A skipped MoE pair must therefore write zeros (or the
MoE reduce must use `select`, not multiply). Writing zeros for the down projection of a dead pair is enough: gate/up of
dead pairs can simply return (their garbage only feeds the dead down pair), the reduce then sums zeros, the shared
expert and all dense ops run on finite inputs.

## 2. What to skip

| part of a cut row, layers > L | skip? | why |
|---|---|---|
| routed experts (gate/up/down, `MUL_MAT_ID`) | yes | ≈ 94 % of the per-row cost at B ≥ 3, cost ∝ live (row, expert) pairs; distinct experts of dead rows are not even read |
| dense (attention / GDN projections, HC mixers, shared expert, router) | no | few-row MMA tiles: B=4→8 costs +0.6 ms in total; skipping needs shape changes |
| attention / GDN core | no | row-coupled; ≈ 0 incremental cost per row |
| router + top-k of dead rows | no | needed to keep shapes; tiny |

Cost of a dead row after the cut: (L+1)/48 · 2.65 + ≈ 0.17 ms instead of 2.82 ms.

## 3. Where it hooks (if GO)

1. **Probe in the graph** (`src/models/qwen4exp.cpp`, after `cb(res_hc, "l_last", il)` at il = L, verify-sized ubatches
   only): score_j from res_hc[j] (linear probe: per-stream RMS norm → dot with a 10240-vector; lens probe: the model's
   final HC mixer + 8·B rows of the output head selected by the MTP candidates). The next-row token is a view of the
   ubatch token input shifted by one, so no host plumbing for it; MTP statistics/candidates (for `linear`/`lens_cand`)
   would be a small per-ubatch input set by the host. Cut with existing ops: `flag = step(score - τ)`,
   `c = cumsum(flag)` (CUMSUM is already on Metal in this model), `live = 1 - step(c - flag)` (exclusive prefix). ≈ 5 small
   dependent ops ≈ 10-20 µs on the critical path, once per pass.
2. **Liveness on the MoE nodes**: in `build_moe_ffn` (llama-graph.cpp) set `src[3] = live` on every `ggml_mul_mat_id` for
   il > L (src[3] is unused by MUL_MAT_ID; the scheduler/allocator treat it as a dependency, so no extra op or barrier per
   layer). Upstream semantics are untouched when src[3] is null.
3. **Metal kernels**: `ggml_metal_op_mul_mat_id` binds src[3] (or a dummy + flag). `kernel_mul_mv_id`: dead pair (token
   iid1 with live = 0) → write zeros to its dst rows and return; in the dedup ballot a dead pair is reported as −1 so the
   owner is the first *live* pair. Same entry test in the q2_0 few-token MoE kernel (0032) and the iq4_nl moe_mt kernel (0020);
   `mul_mm_id` (B > 8) is outside the depth range. The verify kernel list should be taken from a GGML_METAL_PROF_DISPATCH run.
4. **Host**: mark `live` as a graph output and read it after the compute (like `t_h_nextn`), exposed through the staging API
   in `src/llama-ext.h` (e.g. `llama_get_verify_n_live(ctx)`). Server accept block: truncate as in 1.3 before
   `common_sampler_sample_and_accept_n`; pass censoring to the depth controller.
5. Probe weights: a small file (env `LLAMA_ROWPRUNE_PROBE`) loaded into a backend buffer at context creation; τ per depth.

Estimated work if GO: 2-3 days (graph + 3 kernels + host + controller censoring + exactness checks).

## 4. Expected gain [calc]

Cost model (DESIGN 9.14/9.13, current checkpoint): Tv(1) = 17.93 ms, Tv(B ≥ 2) = 23.0 + 2.82·(B−2) ms (B=3 25.83, B=6
34.28), MoE 2.65 ms per live row per pass spread evenly over 48 layers, draft step 1.18 ms, host 0.5 ms. Per-position
acceptance calibrated to the measured tokens per round (prose 2.0 at depth 2, code 4.82 at depth 5):
prose a = 0.65, 0.54, 0.45…; code a = 0.96, 0.91, 0.88, 0.86, 0.85, 0.83, 0.82.

| | base (model) | perfect mid-pass il=15 | il=23 | il=31 | perfect pre-verify truncation | loss per 1 % false drops |
|---|---:|---:|---:|---:|---:|---:|
| prose, depth 2 | 69.8 tok/s | 74.3 (+6.6 %) | 73.1 (+4.8 %) | 72.0 (+3.2 %) | 81.1 (+16 %) | −0.7 % |
| prose, best depth with the cut | 69.8 | 74.3 (depth 2) | | | 87.1 (depth 7, +25 %) | |
| code, depth 5 | 118.6 | 125.0 (+5.4 %) | 123.3 (+4.0 %) | 121.7 (+2.6 %) | 132.5 (+12 %) | −2.2 % |
| code, best depth with the cut | 118.6 | 127.3 (depth 7, +7.3 %) | 124.8 (depth 6, +5.2 %) | | 141.8 (depth 7, +20 %) | −2.8 % (depth 7) |

Realistic: gain ≈ catch × ceiling − false-drop loss. At il=15, prose needs catch ≥ 0.55 at 1 % false drops for +3 %;
code needs catch ≥ 0.6 at 0.5 % false drops for +2 %. Against the +200 % goal this is a small mechanism in every case;
its only strategic value would be making deeper drafts cheap for code.

## 5. Offline measurement (this kit)

- `dumppatch.py` — opt-in dump (`LLAMA_ROWPRUNE_DUMP=<file>`, `LLAMA_ROWPRUNE_LAYERS=15,23,31,47`) in the server: per
  verify pass of slot 0, the HC residual `l_last-<il>` (10240 fp16) of every row at those layers, row tokens, target
  argmax / top-2 / lse / logit of the next row's token per row, accept count, task id, and the MTP top-8 candidates of each
  draft token. Capture through the scheduler eval callback on the target context only (graph split at those nodes: same
  kernels, slower; never time with it). Anchors from 781dbc5 + patches 0001-0046; both patched files pass an MSVC syntax
  check here; not built for Metal, not run.
- `run-dump.sh` — the best-checkpoint switches + the dump; `MODE=fixed7` (default) drafts 7 tokens every round so every
  position is labelled; probe.py truncates records to any depth ≤ 7 (valid by causality, 1.2).
- `collect.py` — sends the MTP-training prompts of `tools/mtp/gendata2.py` (fixed-metric / bench topics excluded) or its
  held-out set; writes a sidecar (kind, text) per request.
- `extract_head.py` — copies `output_hc_{norm,down,up}` (the target's final HC mixer = output norm) out of the main GGUF.
  The output head itself is taken from the MTP GGUF's Q8_0 copy on this PC (probe.py reports the lens/argmax agreement at
  il=47 as a check of the head and of the dump layout; expect ≈ 0.99, not 1.0, since the target uses Q5_K).
- `probe.py` — trains `draft`, `linear_h`, `linear`, `lens_cand`, `lens_full`, `tuned_cand` (and `oracle`), reports AUC,
  catch at 1/3/5 % false drops, and simulated speedups per depth / per kind, standalone and on top of the pre-verify
  truncation by draft confidence ("pre"); prints the go/no-go table. `--selftest` runs the whole pipeline on synthetic dumps.
  Split: train/val by prompt from the training set, test = held-out prompt families (rule 11 of AGENTS.md).
- By-product: the "pre" numbers are an offline evaluation of hypothesis D1 (draft truncation by MTP confidence) on real data.

## 6. Go / no-go

GO to the GPU implementation only if, on the held-out set, some deployable probe (`linear_h`, `linear`, `lens_cand`,
`tuned_cand`; probe cost charged) at il ≤ 23:
1. prose at depth 2: simulated speedup ≥ +3 % and ≥ +2 % over "pre" alone, and
2. code at depth 5: ≥ +2 % and ≥ +1 % over "pre" alone,
3. with lens agreement at il=47 ≥ 0.98 and `nacc == argmax-leading` ≥ 0.999 in the dump (data sanity).
Otherwise NO-GO; record the AUC / catch / speedup table in DESIGN.md as refuted. If only "pre" is strong, that is a result
for D1 (depth control), not for this mechanism.
After a GO the implementation must reproduce ≥ 2/3 of the simulated gain in the fixed metric with identical texts, else revert.

## 7. Risks and unknowns

- The dump changes timing (graph split + syncs + host logit pass); with `MODE=ctl` the depth controller sees other round
  times, hence the fixed-depth default. Outputs should be unchanged — compare a few texts with and without the dump.
- Per-pair independence of the q2_0 and iq4_nl few-token MoE kernels (1.4 ii) is assumed, not verified.
- The MoE cost per row is a slope over B; with masked pairs the launch / ramp per stage stays, so the realised saving may be
  below 2.65 ms × fraction (kbench a masked kernel before building the rest).
- Uniform cost over layers is assumed; actual MoE types differ per layer (iq3_s / iq2_* / q2_0 / iq4_nl, DESIGN 9.1).
- The Q8_0 head of the MTP GGUF stands in for the target's Q5_K head in the lens features.
- One slot only (`-np 1`); replay batches and lookup-drafted rounds are not recorded or have no candidates (flag bit 0).

## 8. Steps on the Mac (do not run during timing measurements)

```bash
# 1. worktree at the best checkpoint + dump patch, build the server only
cd ~/llama.cpp && git worktree add ~/llama-rp mac125b-best-2026-10-11d
mkdir -p ~/rowprune   # copy dumppatch.py collect.py extract_head.py run-dump.sh here (scp from tools/rowprune)
python3 ~/rowprune/dumppatch.py ~/llama-rp            # prints "dumppatch ok"
cd ~/llama-rp && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-server -j 12
# 2. training dump (fixed depth 7), ~120 prompts x 400 tokens, ~9 GB
chmod +x ~/rowprune/run-dump.sh
OUT=~/rowprune/dump-train.bin ~/rowprune/run-dump.sh
python3 ~/rowprune/collect.py --gendata ~/kbench/gendata2.py --set train --n 120 --sidecar ~/rowprune/train.jsonl
~/rowprune/run-dump.sh stop
# 3. held-out dump (prompt families never used for training)
OUT=~/rowprune/dump-held.bin ~/rowprune/run-dump.sh
python3 ~/rowprune/collect.py --gendata ~/kbench/gendata2.py --set held --sidecar ~/rowprune/held.jsonl
~/rowprune/run-dump.sh stop
# 4. the target's final HC mixer (13 MB)
python3 ~/rowprune/extract_head.py ~/models/qwen38-gsq/IQ3_XXS/*.gguf -o ~/rowprune/head.bin
```
(`~/kbench/gendata2.py` is where the MTP tools live on the Mac; otherwise copy `tools/mtp/gendata2.py`.) Copy
`~/rowprune/{dump-train.bin,dump-held.bin,train.jsonl,held.jsonl,head.bin}` to `D:\GPUNW\mac125b\rowprune\`, then on Windows:

```
D:\GPUNW\mac125b\mtptrain\venv\Scripts\python.exe probe.py --dump D:\GPUNW\mac125b\rowprune\dump-train.bin ^
  --sidecar D:\GPUNW\mac125b\rowprune\train.jsonl --held D:\GPUNW\mac125b\rowprune\dump-held.bin ^
  --held-sidecar D:\GPUNW\mac125b\rowprune\held.jsonl --head D:\GPUNW\mac125b\rowprune\head.bin ^
  --mtp-gguf D:\GPUNW\mac125b\mtptrain\mtp-Q8_0.gguf --tuned-steps 2000 --out D:\GPUNW\mac125b\rowprune\report.json
```
