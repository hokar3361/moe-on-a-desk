# qwen4exp QSA selection mask: scatter zeros straight from the raw selected cell indices (opt-out LLAMA_QSA_SIMPLE_MASK=0).
# The reference graph maps every dead slot (padded pool cells = n_kv sentinel, pools invisible to the token) to its own
# dump row so that no cell is written twice. Dead slots either point at the sentinel row (outside the n_kv view) or at
# cells the causal kq_mask already sets to -inf, and every write stores the same 0, so the final mask
# (selected ? kq_mask : -inf) is unchanged while ~16 small dependent ops per QSA layer disappear.
import sys
p = sys.argv[1] + "/src/models/qwen4exp.cpp"
s = open(p, encoding="utf-8").read()
old = """    // live slots address disjoint cells, but padded pools and missing tail cells share the n_kv sentinel, and"""
assert s.count(old) == 1
s = s.replace(old, """    static const bool simple_mask = !getenv("LLAMA_QSA_SIMPLE_MASK") || atoi(getenv("LLAMA_QSA_SIMPLE_MASK")) != 0;
    if (simple_mask) {
        ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));
        GGML_ASSERT(kq_mask->ne[0] == n_kv && kq_mask->ne[1]*kq_mask->ne[2]*kq_mask->ne[3] == n_tokens);
        const size_t row = sel->nb[2];
        sel = ggml_view_4d(ctx0, sel, n_kv, kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3],
                row, row*kq_mask->ne[1], row*kq_mask->ne[1]*kq_mask->ne[2], 0);
        sel = ggml_add(ctx0, sel, kq_mask);
        cb(sel, "indexer_sel", il);
        return sel;
    }

""" + old)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("qsamask ok")
