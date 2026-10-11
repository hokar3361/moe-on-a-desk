# GGML_METAL_PROF_STAGELOG=1: print every encoded node of graphs with > 5000 nodes, "B" when a barrier precedes it
import sys
p = sys.argv[1] + "/ggml/src/ggml-metal/ggml-metal-ops.cpp"
s = open(p).read()
old = """        if (nobar) {
            ggml_mem_ranges_reset(ctx->mem_ranges);
        } else if (!is_concurrent) {"""
new = """        {
            static const bool stagelog = getenv("GGML_METAL_PROF_STAGELOG") != nullptr;
            if (stagelog && ctx->n_nodes_graph() > 5000) {
                fprintf(stderr, "stagelog %c %s %s %s %lld %lld %lld\\n", (!is_concurrent && !nobar) ? 'B' : '.', ggml_op_desc(node), ggml_get_name(node),
                        node->src[0] ? ggml_type_name(node->src[0]->type) : "-", (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2]);
            }
        }
        if (nobar) {
            ggml_mem_ranges_reset(ctx->mem_ranges);
        } else if (!is_concurrent) {"""
assert s.count(old) == 1
s = s.replace(old, new.replace("\\\\n", chr(92) + "n"))
open(p, "w").write(s)
print("stagelog ok")
