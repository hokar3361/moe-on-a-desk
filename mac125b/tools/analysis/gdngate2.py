# register ADD -> SOFTPLUS -> MUL in the fusion table so the graph optimizer keeps the three nodes adjacent
import sys
R = sys.argv[1] + "/ggml/src/ggml-metal/"
p = R + "ggml-metal-fusion.h"
s = open(p).read()
if "GGML_METAL_FUSION_GDN_GATE" not in s:
    old = "    GGML_METAL_FUSION_NORM_MUL_MUL, // RMS_NORM + MUL + MUL (gated norm)"
    assert s.count(old) == 1
    s = s.replace(old, old + "\n    GGML_METAL_FUSION_GDN_GATE,     // ADD + SOFTPLUS + MUL (GDN decay gate)")
    open(p, "w").write(s)

p = R + "ggml-metal-fusion.cpp"
s = open(p).read()
if "ops_gdn_gate" not in s:
    old = "static const std::vector<ggml_op> ops_rms_norm_mul_mul = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL };"
    assert s.count(old) == 1
    s = s.replace(old, old + "\nstatic const std::vector<ggml_op> ops_gdn_gate         = { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL };")
    anchor = "// true if t is or views a tensor in a buffer marked as weights"
    assert s.count(anchor) == 1
    s = s.replace(anchor, """// ADD -> SOFTPLUS -> MUL chain (GDN decay gate), encoded by ggml_metal_op_bin
static bool ggml_metal_fusion_check_gdn_gate(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion); GGML_UNUSED(gf); GGML_UNUSED(node_idxs); GGML_UNUSED(idx); GGML_UNUSED(mode);
    static const bool en = getenv("GGML_METAL_FUSE_GDN_GATE") == nullptr || atoi(getenv("GGML_METAL_FUSE_GDN_GATE")) != 0;
    const ggml_tensor * ad = nodes[0];
    const ggml_tensor * sp = nodes[1];
    const ggml_tensor * ml = nodes[2];
    return en && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS && sp->src[0] == ad && ml->src[0] == sp &&
           ad->type == GGML_TYPE_F32 && ml->type == GGML_TYPE_F32 && ggml_nelements(ad->src[1]) == ad->ne[0] &&
           ggml_nelements(ml->src[1]) == ad->ne[0];
}

""" + anchor)
    old = "    { GGML_METAL_FUSION_NORM_MUL_MUL,   ops_rms_norm_mul_mul,       {},     false, ggml_metal_fusion_check_norm },"
    assert s.count(old) == 1
    s = s.replace(old, old + "\n    { GGML_METAL_FUSION_GDN_GATE,       ops_gdn_gate,               {},     false, ggml_metal_fusion_check_gdn_gate },")
    open(p, "w").write(s)
print("gdngate2 ok")
