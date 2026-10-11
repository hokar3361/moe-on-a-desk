# moe-on-a-desk

Fast single-stream decoding of **Qwen3.8-Flash-Next 125B** (MoE, 512 experts, top-10) on one **Mac Studio M4 Max 64 GB**,
with exact speculative decoding. The quantization is not changed (IQ3_XXS GSQ-RCO, the same file as the baseline, fully
GPU resident); all gains come from new code in a llama.cpp fork, published here as a patch series.

- `mac125b/CONCLUSIONS.md` — what was built, why, and what it gave (conclusions only; read this first)
- `mac125b/DESIGN.md` — the full research log in Japanese (every measurement and trial, including refuted ideas)
- `AGENTS.md` — the working rules of the project (goal, fixed baselines and metric, what counts as a result)
- `mac125b/patches/` — the fork as `git format-patch` output on upstream llama.cpp `781dbc5`
- `mac125b/tools/` — measurement, verification, kernel micro-benchmarks, analysis, MTP training, comparison console

## Result

Fixed metric (`mac125b/tools/measure/mtpab.sh`): same Mac, thinking off, MTP draft head on, decode tok/s from the
llama-server `timings`; bench = `bench-strata.py --lengths 1024,4096 --max-tokens 256 --repeat 3`; code / prose = greedy
400 tokens on one fixed prompt each. The two comparison rows were measured once and are fixed (never re-measured).

| | quantization | bench 1K | bench 4K | code 400 | prose 400 |
|---|---|---:|---:|---:|---:|
| llama.cpp 781dbc5, unmodified, MTP depth 3 (start) | IQ3_XXS + MTP Q8_0 | 76.8 | 75.3 | 76.6 | 48.3 |
| Sushi 1.2.1 (MLX-based) | Sushi-2.6bpw (lower bits) | 85.5 | 83.9 | 90.6 | 62.0 |
| this work `mac125b-best-2026-10-11` | IQ3_XXS + MTP Q8_0 | 94.8 | 93.2 | 99.9 | 60.7 |
| this work `-11d` (mean of 2 runs) | same | 98.4 | 98.8 | 107.0 | 65.9 |
| this work `-11e` (mean of 2 runs) | same | 115.8 | 106.6 | 111.2 | 69.7 |
| **this work `-11f`** (mean of 2 runs) | same | **125.8** | **109.7** | **110.3** | **70.9** |

`-11f` is +64 / +46 / +44 / +47 % over the start. Generality check on 18 held-out prompts (code, poems, reviews; never
used for tuning): start 60.2 → `-11d` 74.5 tok/s; `-11e` +6.3 % over `-11d` and `-11f` +2.2 % over `-11e` (each pair
measured in one session). Reference only, different quantization and engine: TensorFold 0.6.6 with MLX 4-bit on the
same Mac (does not fit in 64 GB, streams experts from SSD) 52.6 / 56.8 / 24.0 / 28.9.

The main mechanisms (details and measured effect of each in `CONCLUSIONS.md` §2): spin wait on GPU completion,
launch-shape fixes, a coarse q1_0 draft head with exact top-32 re-scoring, bounded rollback snapshots, a graph cache per
small-batch shape, delayed frees in the graph allocator, MTP catch-up merged into the first draft step, few-token MoE and
bf16 kernels, a graph reorder and ~10 small-op fusions for the hybrid model (bit-identical), skipping the QSA indexer below
its selection budget, and a **per-draft stop rule** that predicts acceptance from the draft head's own confidence
(held-out AUC 0.888) and drafts only while the expected gain beats the measured cost.

## Reproduce

Hardware / software used: Mac Studio M4 Max 64 GB (40-core GPU), macOS 27.0.1, Xcode command line tools, CMake 3.29.

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

Measure: `mac125b/tools/measure/mtpab.sh "name:ENV=VAL ...|extra server args"` (edit the model paths at the top; it starts and
stops the server itself and writes the answers next to the timings). Single code/prose runs swing by up to ±10 %; use the
mean of at least two runs.

Verify before trusting a build:
1. greedy B=1 output byte-identical to the unmodified build (`tools/measure/verify.sh`-style md5 of 200 tokens);
2. `llama-perplexity -c 512 -b 512 -ub 512` and `-ub 64` **without** `GGML_ALLOC_FREE_DELAY` give the same PPL as the
   unmodified build (catches large-batch bugs that the decode path never exercises).

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

- **Exactness.** The target model verifies every drafted token, so the output is the target's greedy output up to
  numerics: llama.cpp kernels are not batch-invariant, and a different verify width can flip a near-tie, so every
  speculative configuration (including the unmodified start) diverges from one-token-at-a-time greedy decoding after a few
  hundred characters. The code text of the fixed metric is identical across all checkpoints; the prose text changes with
  the drafting policy. Perplexity equals the unmodified build.
- **Fixed bug.** Checkpoints `-11b` to `-11e` contain a fusion that corrupts prompt batches of ≥ 64 tokens when
  `GGML_ALLOC_FREE_DELAY` is not set (PPL 1.23 → 253 at ubatch 512). Every configuration measured here sets it, so the numbers
  stand, but use `-11f` or later (fixed).
- **Comparisons.** Sushi uses a lower-bit quantization (2.6 bpw experts) and its own engine; TensorFold rows are reference
  only (different quantization). Thinking is off for every engine. The stop rule was fitted on 48 separate training prompts;
  the fixed-metric prompts and the held-out set were never used for fitting.
- **Cold pages.** The 27.5 GB n-gram table is read through mmap on the CPU; on a 64 GB machine cold rows cost up to
  4-5 ms per verify. The fixed metric repeats its prompts and sees warm pages.
- One machine, one model. Metal only (the CUDA/CPU paths of the fork are upstream's, untested beyond building).

## In progress (not part of the results above)

The project goal is 3x the start and, next, quantized GLM-5.3 (Q2-Q3) on Mac Studio / RTX 4090-class machines. Current
experiments: a same-model low-bit copy as a remote drafter on a PC GPU while the Mac only verifies (`DESIGN.md` §10,
`mac125b/tools/remote-draft/`). Results will be reported here when measured, against baselines on the same hardware.

## License

The patches modify llama.cpp (MIT License, © the ggml authors). The license of the remaining files is set by the
repository owner.
