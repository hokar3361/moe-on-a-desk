# GGML_METAL_SPLIT_GEO=N0: the main thread encodes only the first N0 nodes and commits them at once (the GPU starts
# almost immediately), helper thread 0 encodes the next N0*GGML_METAL_SPLIT_K nodes, helper thread 1 the rest
# (needs 2 helper threads: GGML_METAL_N_CB=2). Encoding is ~10x faster than the GPU runs the nodes, so each chunk is
# ready before the GPU reaches it.
import sys
p = sys.argv[1] + "/ggml/src/ggml-metal/ggml-metal-context.m"
s = open(p).read()
if "GGML_METAL_SPLIT_GEO" not in s:
    old = """            ctx->n_nodes_0      = MIN(n_main, gf->n_nodes);
            ctx->n_nodes_1      = gf->n_nodes - ctx->n_nodes_0;

            ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;"""
    assert s.count(old) == 1
    new = """            static const int geo   = getenv("GGML_METAL_SPLIT_GEO") ? atoi(getenv("GGML_METAL_SPLIT_GEO")) : 0;
            static const int geo_k = getenv("GGML_METAL_SPLIT_K")   ? atoi(getenv("GGML_METAL_SPLIT_K"))   : 8;
            if (geo > 0 && ctx->n_cb == 2 && gf->n_nodes > geo*(geo_k + 2)) {
                ctx->n_nodes_0      = geo;
                ctx->n_nodes_1      = gf->n_nodes - ctx->n_nodes_0;
                ctx->n_nodes_per_cb = geo*geo_k;   // helper 0: [geo, geo + geo*k), helper 1: the rest
            } else {
            ctx->n_nodes_0      = MIN(n_main, gf->n_nodes);
            ctx->n_nodes_1      = gf->n_nodes - ctx->n_nodes_0;

            ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;
            }"""
    s = s.replace(old, new)
    old = """    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);"""
    assert s.count(old) == 1
    s = s.replace(old, """    if (getenv("GGML_METAL_N_CB")) {
        n_cb = atoi(getenv("GGML_METAL_N_CB"));
    }

""" + old)
    open(p, "w").write(s)
print("geosplit ok")
