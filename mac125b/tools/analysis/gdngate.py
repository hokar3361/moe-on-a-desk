# 1) GDN decay gate ADD(alpha, dt) -> SOFTPLUS -> MUL(., A) in one dispatch (all three outputs written, same expressions)
#    GGML_METAL_FUSE_GDN_GATE=0 disables
# 2) qwen4exp gated norm: build sigmoid(z) before the norm, so RMS_NORM -> MUL(w) -> MUL(gate) are adjacent and fuse
import sys
R = sys.argv[1] + "/"

p = R + "ggml/src/ggml-metal/kernels/unary.metal"
s = open(p).read()
if "kernel_gdn_gate_f32" not in s:
    s += """
// GDN decay gate: v = alpha + dt (broadcast rows), sp = softplus(v), g = sp*A (broadcast rows); every intermediate written
kernel void kernel_gdn_gate_f32(
        constant int2 & nm,              // n elements, row length m (dt and A have m elements)
        device const float * alpha,
        device const float * dt,
        device const float * A,
        device       float * out_add,
        device       float * out_sp,
        device       float * out_mul,
        uint tid[[thread_position_in_grid]]) {
    if ((int) tid >= nm.x) {
        return;
    }
    const int j = tid % nm.y;
    const float v = alpha[tid] + dt[j];
    out_add[tid] = v;
    const float x  = v;
    const float sp = (float) select(log(1 + exp(x)), x, x > 20);
    out_sp[tid]  = sp;
    out_mul[tid] = sp * A[j];
}
"""
    open(p, "w").write(s)

p = R + "ggml/src/ggml-metal/ggml-metal-ops.cpp"
s = open(p).read()
if "GGML_METAL_FUSE_GDN_GATE" not in s:
    old = """int ggml_metal_op_bin(ggml_metal_op_t ctx, int idx) {
    int n_fuse = 1;
    const ggml_metal_fusion * fusion = nullptr;

    const bool use_fusion = ctx->use_fusion();
"""
    assert s.count(old) == 1
    new = old + """
    // GDN decay gate: ADD(alpha, dt) -> SOFTPLUS -> MUL(., A) in one dispatch
    if (use_fusion && idx + 2 < ctx->n_nodes()) {
        static const bool en = getenv("GGML_METAL_FUSE_GDN_GATE") == nullptr || atoi(getenv("GGML_METAL_FUSE_GDN_GATE")) != 0;
        ggml_tensor * ad = ctx->node(idx);
        ggml_tensor * sp = ctx->node(idx + 1);
        ggml_tensor * ml = ctx->node(idx + 2);
        if (en && ad->op == GGML_OP_ADD && sp->op == GGML_OP_UNARY && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS &&
            sp->src[0] == ad && ml->op == GGML_OP_MUL && ml->src[0] == sp &&
            ad->type == GGML_TYPE_F32 && sp->type == GGML_TYPE_F32 && ml->type == GGML_TYPE_F32 &&
            ad->src[0]->type == GGML_TYPE_F32 && ad->src[1]->type == GGML_TYPE_F32 && ml->src[1]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(ad->src[0]) && ggml_is_contiguous(ad) && ggml_is_contiguous(sp) && ggml_is_contiguous(ml) &&
            ggml_are_same_shape(ad, ad->src[0]) && ggml_are_same_shape(sp, ad) && ggml_are_same_shape(ml, ad) &&
            ggml_nelements(ad->src[1]) == ad->ne[0] && ggml_is_contiguous(ad->src[1]) &&
            ggml_nelements(ml->src[1]) == ad->ne[0] && ggml_is_contiguous(ml->src[1])) {
            if (ctx->mem_ranges && (!ggml_mem_ranges_check(ctx->mem_ranges, sp) || !ggml_mem_ranges_check(ctx->mem_ranges, ml))) {
                ggml_metal_op_concurrency_reset(ctx);
            }
            static ggml_metal_pipeline_with_params pl = {};
            if (!pl.pipeline) {
                pl = ggml_metal_library_get_pipeline(ctx->lib, "kernel_gdn_gate_f32");
                if (!pl.pipeline) {
                    pl = ggml_metal_library_compile_pipeline(ctx->lib, "kernel_gdn_gate_f32", "kernel_gdn_gate_f32", nullptr);
                }
            }
            const int nm[2] = { (int) ggml_nelements(ad), (int) ad->ne[0] };
            ggml_metal_encoder_set_pipeline(ctx->enc, pl);
            ggml_metal_encoder_set_bytes   (ctx->enc, (void *) nm, sizeof(nm), 0);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(ad->src[0]), 1);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(ad->src[1]), 2);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(ml->src[1]), 3);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(ad), 4);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(sp), 5);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(ml), 6);
            const int nth = std::min(256, nm[0]);
            ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (nm[0] + nth - 1)/nth, 1, 1, nth, 1, 1);
            return 3;
        }
    }
"""
    s = s.replace(old, new)
    open(p, "w").write(s)

p = R + "src/models/qwen4exp.cpp"
s = open(p).read()
old = """    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);"""
if old in s:
    s = s.replace(old, """    // the gate first, so that RMS_NORM -> MUL(w) -> MUL(gate) are adjacent in the graph and run as one fused kernel
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);
    ggml_build_forward_expand(gf, gated);
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);

    return ggml_mul(ctx0, normalized, gated);""")
    open(p, "w").write(s)
print("gdngate ok")
