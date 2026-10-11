# MoE reduce fusion (MUL + VIEW x n + ADD x (n-1)) for n_expert_used 9..16 (upstream registers 2..8; this model uses 10)
import sys
p = sys.argv[1] + "/ggml/src/ggml-metal/ggml-metal-fusion.cpp"
s = open(p).read()
if "ops_moe_reduce_9" not in s:
    defs = ""
    for n in range(9, 17):
        defs += "static const std::vector<ggml_op> ops_moe_reduce_%d = {\n    GGML_OP_MUL,\n    " % n
        defs += ", ".join(["GGML_OP_VIEW"]*n) + ",\n    " + ", ".join(["GGML_OP_ADD"]*(n-1)) + "\n};\n"
    anchor = "static const std::vector<ggml_op> ops_moe_reduce_8 = {"
    i = s.index(anchor)
    j = s.index("};", i) + 3
    s = s[:j] + defs + s[j:]
    old = "    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     true,  ggml_metal_fusion_check_moe_reduce },\n"
    assert s.count(old) == 1
    new = old
    for n in range(9, 17):
        new += "    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_%d,%s{},     true,  ggml_metal_fusion_check_moe_reduce },\n" % (n, " "*(11 if n < 10 else 10))
    s = s.replace(old, new)
open(p, "w").write(s)
print("moered ok")
