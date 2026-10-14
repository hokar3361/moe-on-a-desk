# moe-on-a-desk

**巨大な MoE モデルを、机の上のコンシューマ機で、量子化を落とさずにどこまで速く動かせるか。**
この 1 台の Mac Studio で得た結果と、そのためのコード(llama.cpp フォークのパッチ)をすべて公開しています。
まだ 1 台でしか試しておらず、推論精度の確認も軽いものです。お手元の環境での結果、品質の検証、改良を、Issue や PR で
ぜひ寄せてください。ローカルで動く AI を、コンシューマ向けの PC 環境でみんなで作っていきましょう。

*How fast can a huge MoE model run on a consumer machine on your desk, without lowering its quantization?* This repository
publishes everything we built for one Mac Studio: the results, the llama.cpp fork as patches, and the full research log.
It has been tested on **one machine only**, and output quality has only been checked lightly — **reports from other
machines, quality evaluations, issues and PRs are very welcome** (see [Help wanted](#help-wanted)).

> **ご利用にあたっての注意(必ずお読みください)**
> - **自己責任でお使いください。** 本リポジトリのコード・手順・数値は無保証です。利用によって生じたいかなる損害についても、作者は責任を負いません。
> - **機材に高い負荷がかかります。** 大きなモデルを動かすため、GPU・メモリ・SSD を長時間ほぼ使い切ります。発熱・消費電力・ファンの騒音が大きくなり、
>   メモリが足りないとほかのアプリが極端に遅くなったり、OS が応答しなくなったりすることがあります。SSD からモデルを読み続ける使い方では SSD の読み出しも多くなります。
>   作業中のデータは保存してから、様子を見ながらお試しください。
> - **llama.cpp を土台にしていますが、改造による副作用は多くの機種では確かめていません。** 動作と数値を確認したのは 1 台の Mac Studio(M4 Max、64 GB)だけです。
>   カーネルのしきい値などはこの GPU に合わせてあり、ほかのチップ・メモリ容量・OS の版では、遅くなる・出力が変わる・動かないといった可能性があります。
>   重要な用途では、未改造の llama.cpp の出力と比べるなどして確認してください。
>
> **Before you use this:** use it at your own risk; it comes with no warranty of any kind. It puts a heavy, sustained load on the GPU,
> memory and SSD (heat, power draw, fan noise; other apps or the OS may become unresponsive when memory runs short). It is based on
> llama.cpp, but the side effects of these modifications have not been checked on other machines: only one Mac Studio (M4 Max, 64 GB)
> was tested, kernel thresholds were tuned for it, and other chips, memory sizes or OS versions may run slower, give different output
> or fail. For anything important, compare with unmodified llama.cpp.

---

## What this is

Single-stream decoding of **Qwen3.8-Flash-Next 125B** (MoE, 512 experts, top-10, ~6B active) with exact speculative
decoding (the model's own MTP draft head; the target verifies every token). The quantization is not changed — the same
IQ3_XXS GSQ-RCO file as the baseline, fully GPU resident — and all gains come from new code.

- `mac125b/CONCLUSIONS.md` — what was built, why, and what it gave (read this first)
- `mac125b/DESIGN.md` — the full research log in Japanese: every measurement and trial, including the ideas that failed
- `AGENTS.md` — the rules the project follows (fixed baselines and metric, what counts as a result, no cheating)
- `mac125b/patches/` — the fork as `git format-patch` output on upstream llama.cpp `781dbc5`
- `mac125b/tools/` — measurement, verification, kernel micro-benchmarks, analysis, MTP training, a comparison console

## Tested on (the only machine so far)

| | |
|---|---|
| Machine | Mac Studio (Mac16,9), Apple **M4 Max** |
| CPU / GPU | 16 cores (12 performance + 4 efficiency) / **40-core GPU**, Metal 4 |
| Memory | **64 GB** unified, 546 GB/s; the GPU working-set limit is about 55.7 GB |
| Storage | 2 TB internal SSD (the model's 27.5 GB n-gram table is read from it through mmap) |
| OS / tools | macOS 27.0.1, Xcode command line tools, CMake 3.29 |
| Power / clocks | default settings; the GPU stays at its top P-state (1578 MHz) during decode |

Nothing else has been measured. Other chips (M1–M5, Pro/Max/Ultra), other memory sizes (32 / 48 / 96 / 128 GB and up)
and other OS versions may behave differently — the kernels were tuned on this GPU, and some thresholds were chosen for it.

## Result

Fixed metric (`mac125b/tools/measure/mtpab.sh`): thinking off, MTP on, decode tok/s from the llama-server `timings`.
bench = `bench-strata.py --lengths 1024,4096 --max-tokens 256 --repeat 3`; code / prose = greedy 400 tokens on one
fixed prompt each. The two comparison rows were measured once on the same machine and are kept fixed.

| | quantization | bench 1K | bench 4K | code 400 | prose 400 |
|---|---|---:|---:|---:|---:|
| llama.cpp 781dbc5, unmodified, MTP depth 3 (start) | IQ3_XXS + MTP Q8_0 | 76.8 | 75.3 | 76.6 | 48.3 |
| Sushi 1.2.1 (MLX-based) | Sushi-2.6bpw (its own quantization) | 85.5 | 83.9 | 90.6 | 62.0 |
| this work `mac125b-best-2026-10-11` | IQ3_XXS + MTP Q8_0 | 94.8 | 93.2 | 99.9 | 60.7 |
| this work `-11d` (mean of 2 runs) | same | 98.4 | 98.8 | 107.0 | 65.9 |
| this work `-11e` (mean of 2 runs) | same | 115.8 | 106.6 | 111.2 | 69.7 |
| this work `-11f` (mean of 2 runs) | same | 125.8 | 109.7 | 110.3 | 70.9 |
| this work `-12a` (mean of 3 runs) | same | 124.7 | 117.9 | 112.4 | 66.8 |
| **this work `-14a`** (mean of 2 runs) | same | **125.4** | **117.7** | **117.1** | **70.4** |
| reference: Strata v0.1.44 + a 64 GB resident fix, MTP `--spec 2` | its own pack from the same IQ3_XXS GGUF | 73.5 | 72.3 | 69.5 | 59.3 |

`-14a` is +63 / +56 / +53 / +46 % over the start. It adds parallel page faults for the cold rows of the 27.5 GB n-gram table on macOS
(llama.cpp's `llama_prefetch` had no Apple implementation): +11 % on never-seen prose, outputs unchanged (`mac125b/CONCLUSIONS.md` §0b).
Strata (an independent engine with a different design, thanks to its author) was measured on the same Mac with the same file and metric;
it is faster one token at a time, this fork at verify widths 2-8 (`mac125b/DESIGN.md` §11). `-12a` was +62 / +57 / +47 / +38 % over the start. It fixes a draft-context bug with prompt-cache reuse
([issue #1](https://github.com/hokar3361/moe-on-a-desk/issues/1)); bench 4K, which repeats its prompts, gains the most.
Measured side by side in one session, `-11f` gave 126.2 / 109.9 / 116.5 / 68.3: code produced the same text, and
code / prose single runs swing by ±10 % (the prose text itself varies between runs of the same build, see below). The other engines are listed as reference points on the same machine,
not as a ranking: each makes different trade-offs (quantization, memory footprint, features), and we learned a lot from them. On 18 held-out prompts (code, poems, reviews; never used for tuning):
start 60.2 → `-11d` 74.5 tok/s; `-11e` +6.3 % over `-11d`, `-11f` +2.2 % over `-11e`. For context, TensorFold 0.6.6
runs the model's MLX 4-bit build on the same Mac by streaming experts from SSD — a capability for models larger than memory
that this setup does not have — at 52.6 / 56.8 / 24.0 / 28.9.

Main mechanisms (each with its measured effect in `CONCLUSIONS.md` §2): spin wait on GPU completion, launch-shape fixes,
a coarse q1_0 draft head with exact top-32 re-scoring, bounded rollback snapshots, a graph cache per small-batch shape,
delayed frees in the graph allocator, MTP catch-up merged into the first draft step, few-token MoE and bf16 kernels, a
graph reorder and ~10 small-op fusions for the hybrid model (bit-identical), exact shortcuts in the QSA sparse attention,
and a **per-draft stop rule** that predicts acceptance from the draft head's own confidence (held-out AUC 0.888) and
drafts only while the expected gain beats the measured cost.

## Second model: Qwen3.6-35B-A3B (2026-10-14)

Asked for by readers who want 26B-35B-class MoE models on smaller Macs: the same fork applied to
**Qwen3.6-35B-A3B** (MoE, 256 experts, top-8, ~3B active, its own MTP head), file `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
(unsloth/Qwen3.6-35B-A3B-MTP-GGUF, 22.9 GB, the MTP block is inside the file), same Mac, same fixed metric.
The baseline is unmodified llama.cpp 781dbc5 with the model's MTP head, and for every column we keep **the best of its
depths 2 / 3 / 4** (the most favourable choice for the baseline).

| | bench 1K | bench 4K | code 400 | prose 400 |
|---|---:|---:|---:|---:|
| llama.cpp 781dbc5, unmodified, MTP, best depth per column | 161.4 | 140.1 | 145.8 | 100.1 |
| this work `moe35-best-2026-10-14a` (mean of 2 runs) | **226.9** | **220.2** | **207.6** | **132.4** |
| | +41 % | +57 % | +42 % | +32 % |

On 18 held-out prompts (not the fixed metric's) it averages 150.4 tok/s against 109.8 (+37 %), and every one of the 18
is faster than the better of the baseline's depth 2 and depth 4 (by 1.28x at least). Three changes made it:

1. **The draft stop rule, retrained for this model** (`mac125b/tools/stop35/`): the rule decides per draft whether one
   more drafted token still pays, from the MTP head's own confidence. Its logistic models, the verify time by width and the
   draft cost are now read from a file (`LLAMA_SPEC_STOP_PARAMS`), fitted on 48 training prompts (held-out AUC 0.883).
   The draft cost was chosen on training prompts only.
2. **Q5_K short rows in the Metal MoE kernel** (patch 0054): the expert down projections have rows of 512 weights (two
   super-blocks), which left half of the SIMD lanes idle. The idle lanes now compute a second group of rows, and their partial
   sums are moved back to the original lane positions before the reduction, so the outputs are bit-identical
   (`GGML_METAL_Q5K_DUAL=0` restores the old dispatch). Verify passes of 4-8 tokens are 15-22 % faster.
3. **A draft-only coarse output head** (patch 0055): each draft step read the full 248k x 2048 Q8_0 output head; now a
   1-bit copy scores the vocabulary and only its top 32 are re-scored with the real rows. The target's verify pass is untouched.

At a fixed draft depth the outputs are identical with and without changes 2 and 3; the code answer is also identical to the
unmodified build. Run it with:

```
GGML_ALLOC_FREE_DELAY=4 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_STOP=1 LLAMA_SPEC_STOP_PARAMS=mac125b/tools/stop35/q36c.params   llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf -ngl 99 -c 16384 --reasoning off --spec-type draft-mtp --spec-draft-n-max 7
```

(`LLAMA_SPEC_MTP_MERGE`, used for the 125B model, lowers this model's draft acceptance and is left off.) The full log is
`mac125b/DESIGN.md` §32. Only this Mac Studio (M4 Max, 64 GB) has been measured; smaller Macs are the next question.

## How much the output quality has been checked (honestly: lightly)

Checked:
- greedy output of the plain decode path (B = 1, 200 tokens) is byte-identical to the unmodified build;
- perplexity equals the unmodified build on our held-out text (c = 512 with ubatch 512 and 64, and c = 6144 where the
  sparse-attention selection is active);
- the code answer of the fixed metric is identical across all checkpoints.

Not checked yet:
- standard benchmarks (MMLU-style, HumanEval-style, long-context retrieval, multilingual, tool use);
- thinking on, sampling with temperature > 0, very long generations, contexts beyond 16k;
- the effect of the verify width: llama.cpp kernels are not batch-invariant, so every speculative configuration
  (including the unmodified start) diverges from one-token-at-a-time greedy decoding after a few hundred characters on
  near-tie tokens. We believe this is harmless, but it has not been measured on a benchmark.

## Help wanted

This was done by one person and an AI agent on one Mac. Things that would help most:

1. **Results from other machines** — open an issue with: machine / chip / GPU cores / memory / macOS, the model files,
   the exact command, and the output of `mtpab.sh` (or simply tok/s from `llama-server` timings), start build vs. this fork.
2. **Quality evaluations** — any benchmark comparing the unmodified build and this fork on the same quantization.
3. **Bugs** — wrong outputs, crashes, memory errors (please include the perplexity check below; it caught one of ours).
4. **PRs** — ports of the ideas to other models or backends, better kernels, better drafting, cleaner code.
   Upstreaming useful parts to llama.cpp is welcome too.

Local AI on consumer hardware gets better when many people try it on many machines. Let's build it together.

## Reproduce

Models (not included; obtain them from their publishers):
- target: ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS** GGUF, 2 shards
  (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf`; 43.8 GiB on the GPU + a 27.5 GB n-gram embedding table
  that stays on the CPU side through mmap)
- draft: the MTP head of Qwen3.8-Flash-Next as a **Q8_0** GGUF sidecar (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, 3.9 GiB).
  The draft-only coarse head is derived from it at load time.

Build:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 781dbc5ac98921dbdb5e5b2ec5b7a50960e937d4
git am /path/to/moe-on-a-desk/mac125b/patches/*.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=OFF
cmake --build build --target llama-server llama-bench llama-perplexity llama-simple -j 12
```

Run (this is `mac125b/tools/measure/run-best.sh`; OpenAI-compatible server on port 8080):

```bash
GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_STOP=1 \
  build/bin/llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 7 --reasoning off \
  -ngl 99 -c 16384 -np 1 --host 127.0.0.1 --port 8080
```

Measure: `mac125b/tools/measure/mtpab.sh "name:ENV=VAL ...|extra server args"` (edit the model paths at the top; it starts
and stops the server and saves the answers next to the timings). Single code / prose runs swing by up to ±10 %; use the
mean of at least two runs.

Verify before trusting a build:
1. greedy B = 1 output byte-identical to the unmodified build (`mac125b/tools/measure/verify.sh`, md5 of 200 tokens);
2. `llama-perplexity -c 512 -b 512 -ub 512` and `-ub 64` **without** `GGML_ALLOC_FREE_DELAY` give the same PPL as the
   unmodified build (this catches large-batch bugs that the decode path never exercises).

### Switches (environment variables of the fork)

| variable | default | meaning |
|---|---|---|
| `LLAMA_SPEC_STOP` | off | per-draft stop rule (the `-11e`/`-11f` configuration; do not combine with `LLAMA_SPEC_DEPTH_CTL`) |
| `LLAMA_SPEC_STOP_LAM`, `_LAM_A` | 0.09, 0.01 | initial long-run rate (tokens/ms) and its EMA weight per round |
| `LLAMA_SPEC_DEPTH_CTL` (+ `_MIN`, `_RESET`) | off | the older per-round depth controller (`-11d` and earlier) |
| `LLAMA_SPEC_MTP_MERGE` | off | MTP catch-up merged into the first draft step |
| `LLAMA_GRAPH_CACHE` | 0 | graphs kept per small-batch shape |
| `GGML_ALLOC_FREE_DELAY` | 0 | reuse memory only K nodes after its last reader (fewer false barriers) |
| `LLAMA_MTP_COARSE_HEAD` | q1_0 | type of the draft-only coarse head (`off` disables) |
| `LLAMA_QSA_DENSE_SHORT`, `LLAMA_QSA_SIMPLE_MASK` | on | exact QSA shortcuts (below / above the selection budget) |
| `GGML_METAL_SPIN_WAIT` | on | poll the command buffer instead of a blocking wait |
| `GGML_METAL_REORDER_EXT`, `GGML_METAL_FUSE_*`, `GGML_METAL_TOPK_REG` | on | graph reorder over the hybrid model's ops, bit-identical fusions, register top-k |
| `GGML_METAL_TIMING`, `GGML_METAL_PROF_*` | off | measurement only (some of them produce wrong outputs on purpose) |

## Caveats

- **One machine, one model, Metal only.** The CUDA / CPU paths of the fork are upstream's and were only built, not tuned.
- **Fixed bug.** Checkpoints `-11b` to `-11e` contain a fusion that corrupts prompt batches of ≥ 64 tokens when
  `GGML_ALLOC_FREE_DELAY` is not set (PPL 1.23 → 253 at ubatch 512). Every configuration measured here sets it, so the
  numbers stand, but use `-11f` or later.
- **Fixed bug ([issue #1](https://github.com/hokar3361/moe-on-a-desk/issues/1)).** Up to `-11f`, `LLAMA_SPEC_MTP_MERGE=1`
  could flush the deferred MTP catch-up of the previous request into the draft context when a new request reused a cached
  prompt prefix, so the first draft of each such request failed (`inconsistent sequence positions`, `deferred catch-up
  failed`). The target output is not affected (drafts are always verified), only speed. Fixed in `-12a` (patch 0051).
- **Comparisons.** The engines use different quantizations (Sushi: its own 2.6 bpw expert format; TensorFold: MLX 4-bit),
  so the numbers are reference points, not a like-for-like ranking. Thinking is off for every engine. The stop rule was fitted on 48 separate training
  prompts; the fixed-metric prompts and the held-out set were never used for fitting.
- **Cold pages.** The 27.5 GB n-gram table is read through mmap on the CPU; on a 64 GB machine cold rows cost up to
  4-5 ms per verify. The fixed metric repeats its prompts and sees warm pages; a first-time text can be slower.

## What's next

- 3x the start on this Mac (the project goal) — the ceiling analysis is in `CONCLUSIONS.md` §4.
- **GLM-5.3-Flash** (320B MoE, Q2-Q3) on a 64 GB Mac and an RTX 4090-class PC, including using both machines together.
  Baselines on the same hardware first; results will appear here.

## Acknowledgements

This work stands on the shoulders of the local-LLM community. Thank you to:
- **llama.cpp / ggml** (ggml-org and contributors) — the engine this fork builds on, including the model support, the MTP
  speculative path and the Metal backend;
- **Qwen** for Qwen3.8-Flash-Next, **ISTA-DASLab** for the GSQ-RCO quantization, and the authors of the MTP GGUF sidecar;
- **Sushi** — a fast MLX-based engine for the same model; its write-ups on few-row expert kernels matched and guided our own
  findings, and its idea of warming the n-gram table is on our list;
- **TensorFold** — SSD expert streaming that makes models larger than memory usable on Macs, and a reference we measured;
- **Strata** — `bench-strata.py` in `mac125b/tools/measure/` comes from the Strata project and is the benchmark we use;
- **MLX**, **mlx-flash** and everyone sharing measurements, kernels and ideas in public.

Every one of these projects pushes local AI forward. If we misrepresent any of them, please open an issue and we will fix it.

## License

The patches modify llama.cpp (MIT License, © the ggml authors). The license of the remaining files is set by the
repository owner.
