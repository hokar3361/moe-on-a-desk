# mac125b — Conclusions

Single-stream decode of Qwen3.8-Flash-Next 125B (MoE, 512 experts top-10, ~6B active; hybrid Gated-DeltaNet +
QSA attention; 4-stream hyper-connections; MTP head) on one Mac Studio M4 Max 64 GB (546 GB/s), quantized
IQ3_XXS (GSQ-RCO, 43.8 GiB, fully GPU resident), with exact speculative decoding (MTP draft head, greedy output
identical to the non-speculative model).

This file keeps **conclusions only** (what we changed, why, and what it gave), for the current best checkpoint.
The full measurement log and every trial live in `DESIGN.md`. When a conclusion is superseded, the old version is
reduced to a one-line summary here and the new conclusion takes its place.

## 0. Result (fixed metric, same Mac, thinking off, MTP on; decode tok/s)

Checkpoint: fork tag `mac125b-best-2026-10-14a` (`mac125b/patches/` on upstream llama.cpp `781dbc5`; run with
`mac125b/tools/measure/run-best.sh`). Earlier checkpoints: `-12a` 124.7 / 117.9 / 112.4 / 66.8 (mean of 3), `mac125b-best-2026-10-11` 94.8 / 93.2 / 99.9 / 60.7, `-11b` 97.6 / 95.7 / 101.3 / 62.6, `-11c` 98.3 / 96.5 / 102.9 / 63.3,
`-11d` 98.4 / 98.8 / 107.0 / 65.9, `-11e` 115.8 / 106.6 / 111.2 / 69.7, `-11f` 125.8 / 109.7 / 110.3 / 70.9.

| | Model | bench 1K | bench 4K | code 400 | prose 400 |
|---|---|---:|---:|---:|---:|
| Start: llama.cpp 781dbc5, unmodified | IQ3_XXS + MTP Q8_0 | 76.8 | 75.3 | 76.6 | 48.3 |
| Sushi 1.2.1 (MLX-based) | Sushi-2.6bpw | 85.5 | 83.9 | 90.6 | 62.0 |
| **This work** `-14a` (mean of 2 runs) | same as start | **125.4** | **117.7** | **117.1** | **70.4** |

+63 / +56 / +53 / +46 % over the start (`-14a`). `-12a` was +62 / +57 / +47 / +38 % (`-11f` side by side in the same session: 126.2 / 109.9 / 116.5 / 68.3; `-12a` is a bug fix, see §5; prose single runs swing 62-71); Sushi (its own 2.6 bpw quantization) is listed as a reference point on the same machine, not as a ranking. Code runs swing ±10 % (115.7 and 104.9). Target: +200 %.
The code text is identical to the first checkpoint's; the prose text diverges at character 244. Every speculative configuration,
including the earlier checkpoints, diverges from plain one-token greedy decoding after 200-300 characters: the verify width changes
the rounding of near-tie logits. This is not introduced by the stop rule. Reference only (different quantization): TensorFold 0.6.6
on the same Mac, MLX 4-bit (does not fit 64 GB, streams experts from SSD): 52.6 / 56.8 / 24.0 / 28.9.

## 0b. Cold rows of the n-gram embedding table on macOS (`-14a`)

Observation: without speculation, the first answer on a new topic ran 8 % slower than the same answer again (50.1 vs 54.3 tok/s);
the fixed-metric texts, already warm, did not show it. The PLE n-gram table (27.5 GB, read lazily from the mmap row by row) is
prefetched by `llama_prefetch`, which only had Linux (madvise) and Windows implementations: on macOS it did nothing, so each cold
row was a page fault taken one at a time inside get_rows. Change: on Apple platforms, fault the pages of the requested rows in
concurrently (GCD `dispatch_apply`) before the gather; reads only, outputs unchanged (byte-identical one-token outputs with it on
and off; the 200-token md5 check passes). Effect: on eight never-seen story prompts, MTP configuration 64.9 → 71.9 tok/s (+11 %),
without speculation 51.0 → 53.4 (+4.7 %); fixed metric 125.4 / 117.7 / 117.1 / 70.4 (two runs).

## 1. Where the time goes (the facts every design decision below rests on)

1. **A speculative round is 86 % verify pass.** Depth 3, prose: 40.0 ms = verify 34.3 + 3 draft steps 4.0 + MTP
   catch-up 1.24 + host gaps 0.5 ms.
2. **The verify pass is ~2.3x its bandwidth floor at every width.** T(B) ≈ 20.6 ms + ~2.8 ms per extra token (B ≥ 4);
   the floor is 8.6 ms + ~1.2 ms per extra token (bytes of weights, distinct experts per token).
3. **The GPU kernels are occupancy/latency bound, not bandwidth or ALU bound.** A MoE matvec costs the same with its
   weights resident in the SLC as from DRAM; cutting its decode instructions by ~40 % gave 4 %; cutting live registers
   gave 10 %. (The Sushi authors record the same for their own kernels.)
4. **Dense few-token matmuls pay a fixed premium.** For B = 2..4 the few-row MMA path costs the same at every B and
   20-100 % more than the B = 1 matvec (q4_K 32 → 54-64 µs), i.e. the extra tokens are not what costs.
5a. **Stage serialisation is the largest single block of the verify pass, and its floor is the heavy-op drain.** Dropping every barrier (timing only) takes B=3 from
   30.2 to 19.1 ms; most of it is the drain and DRAM ramp of a heavy op before a dependent small op (4-12 µs per heavy op).
   After removing 700 small stages (B=3 29.6 → 26.9 ms) the barriers before small ops still cost 4.3 ms: every heavy op on
   the critical path is followed by a dependent op, so with one kernel per op this drain cannot be hidden.
5. **The SoC lowers fabric clocks when the host CPU sleeps** during a GPU wait: a blocking wait costs ~10 % of decode.
6. **Draft acceptance falls with depth** (code: 0.95 first position, 0.66-0.75 deeper; prose ~0.6 then ~0.45).
   The recursive MTP state is far from the target state it replaces (per-stream cosine 0.48), yet it predicts
   better than any linear correction towards the target state.
7. **The MoE kernels are bound by their instruction stream, not by memory.** Removing every weight and activation read
   from the q2_0 MoE kernel keeps 81-86 % of its time (B=1 17.1 → 13.9 µs, B=4 56.0 → 48.1 µs; launch + grouping alone
   4.8 / 7.5 µs). The arithmetic (decode, multiply-add, pre-scaling, reduction) and its dependency stalls dominate, and it is
   redone per token, so it grows linearly with the verify width.
8. **A 27.5 GB n-gram embedding table lives outside the GPU.** `per_layer_token_embd` (IQ4_NL, 160 × 320 M rows) is read on
   the CPU, 16 rows per token, through mmap; on 64 GB it cannot all stay in the page cache. Warm rows cost 0.07 ms per verify,
   cold rows up to 4-5 ms (SSD page faults). The fixed metric repeats its texts and sees the warm case.

## 2. Mechanisms in the checkpoint (all new code in the fork; greedy output identical to the baseline)

| # | Mechanism | Observation that motivated it | What it does | Effect |
|---|---|---|---|---|
| M1 | Spin wait on GPU completion | §1.5 | poll the command buffer status instead of sleeping | +10 % decode |
| M2 | Launch-shape fixes (SSM_CONV, unary/binary row kernels) | ~400 k one-thread threadgroups per token | one thread per element in full groups | +4 % |
| M3 | Coarse draft head with exact rescoring | the 248k-vocab head dominated a draft step | q1_0 copy of the head ranks the vocabulary, the top-32 rows are re-scored with the real Q8_0 head (draft only; verify uses the real head) | draft step 2.04 → 1.18 ms, acceptance unchanged |
| M4 | Bounded rollback snapshots | snapshot copies grew with the draft cap | copy only the slots a batch can roll back | deep drafts no longer slow the verify |
| M5 | Adaptive draft depth | best depth differs by content (code deep, prose 2) | model-based: expected accepted tokens from per-position acceptance ÷ measured round time per depth; reset per request; explores one deeper position often when the last position is usually accepted | code / bench +10-20 % over the best fixed depth, prose within ~2 % of depth 2 |
| M6 | Graph cache per small-batch shape | depth switching rebuilt the 7k-node verify graph (~3 ms each) and the MTP catch-up/draft graphs every round | keeps a scheduler + allocated graph per ubatch shape (LRU) | controller prose 53.8 → 59.2 |
| M7 | Delayed frees in the graph allocator | freeing a tensor right after its last reader lets the next node overwrite it while that reader may still run → false write-after-read barriers | memory is reused only K = 4 nodes after the last reader | barriers 2129 → 1999 per verify; depth-3 bench +6 % |
| M8 | MTP catch-up merged into the first draft step | the catch-up ran the MTP head over every verified token, rejected ones included, as its own decode | the next draft decodes the accepted prefix and the new token in one forward | −0.44 ms per round |
| M9 | Few-token q2_0 MoE kernel | q2_0 down experts used a select-based decode with 62.5 % lane utilisation | half2 "magic" decode (`(w & mask) | 0x6400` read as half), 8 lanes × 4 rows, K split exactly | −21 % (B=1) / −25 % (B=4) per op |
| M10 | bf16 few-row MMA from 2 rows | upstream threshold (4 rows, tuned on M3 Ultra) left B = 3 on a slow path | threshold 2 on M4 Max | B=3 verify 33.9 → 30.3 ms; prose depth 2 +5 % |
| M11 | Graph reorder over the hybrid model's ops (after the tag) | upstream reorder stops its look-ahead at any op outside a short "safe" list; this model is full of CONCAT, SOFT_MAX, HC_PRE/POST, TOP_K, GATED_DELTA_NET, FLASH_ATTN... | those ops (memory access fully described by src/dst) may be moved into earlier concurrent stages | barriers 1897 → 1768, B=3 verify −1.0 ms |
| M12 | MoE reduce for 10 experts, + shared-expert add (after the tag) | upstream fuses the expert weighting+sum only up to 8 experts; with 10 it splits into 4 dependent stages | pattern up to 16 experts; product rounded alone (`precise::fma(e,w,0)`) so the sum is bit-identical; the following ADD of the shared expert in the same dispatch | B=3 verify −0.6 ms |
| M14 | QSA below its selection budget | when the indexer would select every pool (context ≤ top_k ≈ 2k tokens) its mask equals the causal mask | skip the indexer query, scoring, top-k and the ~15-stage mask build, keep the indexer cache updates | barriers 1282 → 1025, verify −1.05 ms at every width, bit-identical |
| M15 | Depth-controller round-time fixes | the first round of every request (spanning prompt processing) was stored as the round time of a never-measured depth (e.g. 18.8 ms for depth 4), so prose ran too deep | skip that round, accept a first measurement only near the prior, keep T(d) monotone, priors from the current verify costs | prose 61.9 → 65.9, bench 4K +2 %, stable run to run |
| M16 | Per-draft stop rule from the draft head's own confidence (replaces the per-round depth controller M5/M15 in the checkpoint) | logged 7,500 rounds at depth 7 on 48 training prompts + 18 held-out prompts: whether draft k is accepted is predicted with AUC 0.888 by the MTP head's top-1 probability, logit margin and weakest earlier draft, but only 0.656 by position + acceptance history (what the depth controller used) and 0.683 by the verifier's own signals | logistic per position (fitted on training prompts only), after each draft step: keep the draft and continue only while the expected accepted tokens of the best look-ahead exceed λ × (verify growth + draft steps), λ = realised tokens per ms as a slow EMA (0.01 per round; tracking it fast stopped drafting right after good rounds, −1.6 points in replay); before a step, the same test with a model of the next draft from the previous one | fixed metric 98.4 / 98.8 / 107.0 / 65.9 → 115.8 / 106.6 / 111.2 / 69.7; held-out 18 prompts +6.3 %, all 18 faster (+0.9 to +19 %) |
| M17 | QSA selection mask from the raw selected cells; register bitonic top-k (after -11e) | above the selection budget (context > ~2k) each QSA layer built its mask through ~16 dependent small ops that give every dead slot its own dump row, and the 1024-wide bitonic top-k re-read both scores from device memory on all 55 barrier steps | dead slots point either at the sentinel row outside the mask or at cells the causal mask already sets to −inf, and every write stores 0, so scatter the raw indices directly; top-k with the (score, index) pair in registers, partners within a SIMD group by shuffle (same network, same swaps) | 4K context verify B=1 19.28 → 18.86, B=3 27.93 → 27.35 ms; PPL at 6k context identical to the start (1.3402) |
| M13 | Small-op fusions around the hyper-connection mixers and the GDN block | tiny dependent ops between the HC matrices and in the GDN gated norm | HC_POST + grouped RMS norm + gamma; SCALE+SILU applied on the src1 load of the bf16 up matrix whose MMA reads rows as 4·e+stream and computes the gated HC_PRE from the reduced tile (the HC mix goes from 7 stages to 3); gated RMS norm (sigmoid built first so the 3 nodes fuse); GDN decay gate ADD+SOFTPLUS+MUL. Same threads, expressions and reduction order as the unfused kernels: bit-identical | barriers 1768 → 1282, B=3 verify −1.2 ms |

## 3. Conclusions on directions that do not pay (summaries; details in DESIGN.md)

- Persistent megakernel / stage removal: replaying the real stage sequence with ideal kernels shows stages cost only
  ~1.2 ms of a 34 ms verify; the waste is inside the kernels.
- Prefetching the next stage's weights (same or second queue): Metal barriers wait for every prior dispatch and two
  queues serialize; it adds time.
- MoE kernels that share work across the tokens of a verify batch: tokens share only ~30 % of their experts and the
  extra live state lowers occupancy (tiled, grouped, lane-per-row designs all slower).
- Codebook decode tricks (float tables, xor sign tables, wide threadgroup loads): 16-byte threadgroup loads are 12x
  slower than 4-byte ones on M4 Max; only the half2 magic decode is a win, and only with fewer live registers.
- Re-laying IQ blocks for wide aligned loads: the access pattern alone already runs near the floor; no gain.
- Lossless structure in the bf16 hyper-connection weights: none usable (stream blocks not proportional, full rank);
  exponent entropy coding would save at most ~2 %.
- Making the recursive MTP state look like the target state (ridge-regressed linear map): closer in L2, lower
  acceptance. Recursive fine-tuning of the MTP block on 0.3 M self-generated tokens: +2.5 / +4.9 points at draft
  positions 3 / 4 offline, no measurable speedup. The offline gain was an artefact: the evaluator built the KV of all past positions
  from the recursive state, while inference uses first-stage KV for accepted positions. With an evaluator matching inference
  (accepted tokens per round within 0-4 % of the machine) that head is 1 point below the original, and retraining with
  inference-matched KV on 0.27 M tokens moves positions 2-4 by −0.6 to +0.1 points. Closed at this data scale.
- Computing some experts on the idle CPU during verify: 12 threads reach 33-52 GB/s per expert (9-16 µs each), so the CPU
  could take 3-4 of ~28 experts per layer, ≤ 4-6 % of verify before two CPU↔GPU synchronisations per layer.
- Activation reads in the MoE kernel: removing them saves 4-5 %.
- q4_K few-row MMA with a q4_K-specific decode (−22 % per op, verify −1 %): withdrawn, it changes the verify-width numerics
  so the fixed-metric prose text changed; not worth a non-identical output for 1 %.
- Prefetching the next matrix's weights into the SLC from DRAM-idle stages (small ops, MoE stages): every variant slower.
- Rewriting the IQ2 MoE kernels with a cheaper decode and per-expert token grouping: slower (24.7 → 28.9 µs at B=1);
  grouping serialises the tokens of an expert inside one threadgroup and B=8 has only ~1.4 tokens per expert.
- Keeping the round times across requests (opt-in `LLAMA_SPEC_DEPTH_KEEP_T`): bench 1K/4K 115 / 116 but prose 62 (it starts
  prose at depth 7 from times learned on code); climbing one depth at a time fixes neither. Superseded by the per-draft stop rule (M16), which gets the bench gain without the prose loss.
- Persistent kernels for dependent bf16 matvec chains: −10 % per op at best (15.2 vs 16.8 µs), deadlock above ~240 resident groups.
- Outside-the-kernel causes of the slow verify pass are ruled out: the GPU stays at its top P-state (1578 MHz) for 93-99 % of decode
  (IOReport residency), and copying the GPU weights into Metal-owned buffers instead of wrapping the mmap (16 KB pages) changes nothing
  (B=1 18.26 → 18.19-18.25 ms, B=3 26.00 → 25.99 ms). The loss is inside the kernels (latency, occupancy, stage drain).
- A persistent draft kernel: the whole 167-node draft graph without any barrier is only 0.13 ms faster (1.06 → 0.93 ms GPU).
- Bayesian per-position acceptance with round times kept across requests (opt-in `LLAMA_SPEC_DEPTH_BAYES`): unstable from run to run
  (83.7 / 77.6 / 77.6 / 59.8 and 107.9 / 103.1 / 102.7 / 64.8 with every measurement guarded); dropped.
- Co-scheduling the next round's drafts inside the verify stages works on the GPU, but only speculative drafts are
  independent of the verify result; expected gain ≤ 5 %.

Generality (reference measurement, 18 held-out prompts never used for training or the fixed metric, greedy 400 tokens):
start 60.2 → -11d 74.5 tok/s (1.24x; code 1.26x, poems 1.22x, reviews 1.18x). -11e vs -11d in one session: 79.2 → 84.2 (+6.3 %),
every prompt faster (code +1 to +9 %, poems +4 to +10 %, reviews +5 to +19 %). The gains are not prompt-specific. The stop rule
was fitted on 48 separate training prompts; the held-out set was only used to score it.
Prose-like text gains slightly less because it accepts ~2 tokens per round, so the fixed cost of a verify pass dominates.

## 4. Ceiling under the rules (computed from measurements)

With every kernel at the ideal of the replay experiment (pure weight streaming, launch + barrier per stage: 11.63 ms for
one token, +1.29 ms per extra token), drafts at 0.6 ms and 0.3 ms host time per round, and today's acceptance:

| | depth, tokens per round | ideal round | ceiling | 3x target |
|---|---|---:|---:|---:|
| prose | 2, 2.0 | 15.7 ms | ~127 tok/s (~138 with a persistent megakernel) | 145 |
| code | 5, 4.82 | 21.4 ms | ~225 tok/s | 230 |

Kernel work alone cannot reach 3x. It needs both near-ideal kernels (verify at about half of today's time; the dedicated
Sushi engine, faster than ours on this pass, takes 0.87x of our time) and more accepted tokens per round (prose +14-20 %).

## 5. Correctness fix (2026-10-11)

The HC_POST + grouped-norm fusion (M13, since `-11b`) read the block output shared by the four streams from every threadgroup, but
its second output could be allocated in that block output's memory once it was dead. With the allocator's delayed frees
(`GGML_ALLOC_FREE_DELAY=4`, used by every server configuration we measured) this never happened, so the fixed-metric numbers stand;
without it, prompt batches of ≥ 64 tokens were corrupted (PPL 1.23 → 253 at ubatch 512). The fusion now refuses outputs that overlap its
inputs. Since then every checkpoint is also checked for perplexity equal to the start build without delayed frees (c = 512, ubatch 512 and 64).

## 5b. Correctness fix (2026-10-12, GitHub issue #1)

Observation: a user running consecutive chat requests that share a prompt prefix saw `inconsistent sequence positions` in the
MTP draft context on the first draft of every request, and `deferred catch-up failed`. Reproduced on the same Mac (9 errors in
10 requests); the unmodified build, the fork without env vars and the fork without `LLAMA_SPEC_MTP_MERGE` gave 0.
Cause: M8 (merged catch-up) deferred the catch-up of any small batch while a "generating" flag was set, and cleared the flag only
in begin(), which the server calls after the prompt. The last few prompt tokens of a new request (split off for the recurrent
checkpoint) are such a small batch, so the previous request's deferred rows were flushed into the draft context at their old
positions. Change: only the verify batch of the last draft (it starts at that draft's pos0) is deferred; any other batch drops a
pending deferral; a flush never writes rows that do not continue the draft context. Effect: 0 errors; the target output is
unchanged (drafts are verified; the code text is identical); bench 4K, which repeats its prompts, 109.9 → 117.9 tok/s.

## 6. Quality (2026-10-13, same Mac, greedy, thinking off)

KL divergence against the start build's own ubatch-512 logits (wikitext-2, 40 x 512): ub 1 start 0.00748 / `-12a` 0.00742,
ub 4 start 0.00690 / `-12a` 0.00693, top-1 agreement equal (97.28 % / 97.45 %), ub 512 bit-identical. Accuracy: GSM8K-200 97.0 % / 97.0 %,
MGSM-ja-250 86.4 % / 86.4 %, GSM8K 4-turn conversations 98.5 % / 98.0 % (one question), at 1.4x the decode speed.
The unmodified CUDA build on another GPU gives 97.0 % / 86.8 %.
