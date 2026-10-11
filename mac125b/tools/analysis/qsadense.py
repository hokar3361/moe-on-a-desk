# QSA: when the indexer would select every pool (n_pool <= top_k/kpool, i.e. the context fits the selection budget),
# the selection mask equals the causal mask: skip the indexer query, scoring, top-k and mask construction, keep the
# indexer cache updates (raw keys and the pooled keys of completed blocks) for later, longer contexts.
# LLAMA_QSA_DENSE_SHORT=0 disables.
import sys
p = sys.argv[1] + "/src/models/qwen4exp.cpp"
s = open(p).read()
if "LLAMA_QSA_DENSE_SHORT" not in s:
    old = """    ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
    ggml_tensor * pooled = kpool_cache.gather_pooled(inp_kpool->pool_cells);"""
    assert s.count(old) == 1
    new = """    ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));

    // every pool selected: the selection mask is the causal mask (selected cells add 0, the rest are masked anyway)
    {
        static const bool dense_short = getenv("LLAMA_QSA_DENSE_SHORT") == nullptr || atoi(getenv("LLAMA_QSA_DENSE_SHORT")) != 0;
        if (dense_short && n_pool <= (int64_t) (hparams.indexer_top_k / kpool)) {
            return kq_mask;
        }
    }

    ggml_tensor * pooled = kpool_cache.gather_pooled(inp_kpool->pool_cells);"""
    s = s.replace(old, new)
    open(p, "w").write(s)
print("qsadense ok")
