# llama.cpp fork as a patch series

Base: llama.cpp `781dbc5` ("spec: properly handle mtmd input for mtp (#30257)").
The patches here are the fork up to the best checkpoint `mac125b-best-2026-10-11d` (fork commit b6656a2). Patches 1-35 are the
first checkpoint (`mac125b-best-2026-10-11`, 12d8306); the later ones add the fewer-stage verify pass (graph reorder over the hybrid
ops, MoE reduce for 10 experts + shared-expert add, SCALE+SILU, gated RMS norm, HC_POST + grouped RMS norm, HC mix tail in one MMA dispatch, GDN gate), QSA below its selection budget, depth-controller fixes, opt-in experiments
and profiling switches.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 781dbc5
git am /path/to/mac125b/patches/*.patch
```

Mechanisms are switched by environment variables (see `../CONCLUSIONS.md` and `../tools/measure/run-best.sh`);
experimental/refuted ones stay opt-in and off by default.
