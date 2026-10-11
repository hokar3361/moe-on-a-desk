# GGML_METAL_PROF_NOBAR_OPS=op1,op2,... (timing only, results are wrong): drop the barrier that a node of one of these
# ops would force (ranges are still cleared). "MM" = MUL_MAT and MUL_MAT_ID, "SMALL" = every op that is not a matmul.
import sys
p = sys.argv[1] + "/ggml/src/ggml-metal/ggml-metal-ops.cpp"
s = open(p).read()
old = """        if (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
            // GGML_METAL_PROF_BARLOG"""
new = """        static const char * nobar_env = getenv("GGML_METAL_PROF_NOBAR_OPS");
        bool nobar = false;
        if (!is_concurrent && nobar_env) {
            const bool mm = node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID;
            const std::string list = std::string(",") + nobar_env + ",";
            nobar = (mm && list.find(",MM,") != std::string::npos) || (!mm && list.find(",SMALL,") != std::string::npos) ||
                    list.find(std::string(",") + ggml_op_desc(node) + ",") != std::string::npos;
        }
        if (nobar) {
            ggml_mem_ranges_reset(ctx->mem_ranges);
        } else if (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
            // GGML_METAL_PROF_BARLOG"""
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w").write(s)
print("nobarops ok")
