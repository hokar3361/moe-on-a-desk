# MoE reduce + the following ADD (moe_out + shared expert) in one dispatch; both outputs written, same arithmetic
# GGML_METAL_FUSE_MOE_ADD=0 disables
import sys
R = sys.argv[1] + "/ggml/src/ggml-metal/"

p = R + "kernels/argsort.metal"
s = open(p).read()
if "kernel_moe_reduce_add_f32" not in s:
    s += """
// MoE reduce and the following residual-style ADD: dst = sum_e(experts[e]*weights[e]); dst2 = dst + other
kernel void kernel_moe_reduce_add_f32(
        constant   ggml_metal_kargs_moe_reduce & args,
        device const float * experts,
        device const float * weights,
        device       float * dst,
        device const float * other,
        device       float * dst2,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int64_t token = tgpig.x;
    const int64_t col   = (int64_t) tgpig.y * ntg.x + tpitg.x;
    if (token >= args.ne02 || col >= args.ne00) {
        return;
    }

    const int n_expert_used = FC_moe_reduce_n_expert_used;

    const int64_t base = token * (int64_t) n_expert_used * args.ne00 + col;
    float sum = 0.0f;
    FOR_UNROLL (int e = 0; e < n_expert_used; ++e) {
        sum = sum + metal::precise::fma(experts[base + e * args.ne00], weights[token * n_expert_used + e], 0.0f);
    }
    dst[token * args.ne00 + col] = sum;
    dst2[token * args.ne00 + col] = sum + other[token * args.ne00 + col];
}
"""
    open(p, "w").write(s)

p = R + "ggml-metal-device.cpp"
s = open(p).read()
if "kernel_moe_reduce_add_f32" not in s:
    anchor = "ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce(ggml_metal_library_t lib, int32_t n_expert_used) {"
    assert s.count(anchor) == 1
    s = s.replace(anchor, """ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce_add(ggml_metal_library_t lib, int32_t n_expert_used) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_moe_reduce_add_f32");
    snprintf(name, 256, "%s_n_expert_used=%d", base, n_expert_used);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int32(cv, n_expert_used, FC_MOE_REDUCE + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

""" + anchor)
    open(p, "w").write(s)

p = R + "ggml-metal-device.h"
s = open(p).read()
if "get_pipeline_moe_reduce_add" not in s:
    anchor = "struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_used);"
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + "\nstruct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce_add    (ggml_metal_library_t lib, int32_t n_expert_used);")
    open(p, "w").write(s)

p = R + "ggml-metal-ops.cpp"
s = open(p).read()
old = """    auto pipeline = ggml_metal_library_get_pipeline_moe_reduce(lib, (int32_t) experts->ne[1]);

    const int nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int n_col_tiles = (args.ne00 + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(experts), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),     3);
"""
assert s.count(old) == 1
new = """    // the next node adds another tensor to the reduced output (MoE out + shared expert): one dispatch writes both
    ggml_tensor * add = nullptr;
    {
        static const bool en = getenv("GGML_METAL_FUSE_MOE_ADD") == nullptr || atoi(getenv("GGML_METAL_FUSE_MOE_ADD")) != 0;
        if (en && idx + n_fuse < ctx->n_nodes()) {
            ggml_tensor * nx = ctx->node(idx + n_fuse);
            if (nx->op == GGML_OP_ADD && nx->type == GGML_TYPE_F32 && (nx->src[0] == dst) != (nx->src[1] == dst)) {
                const ggml_tensor * oth = nx->src[0] == dst ? nx->src[1] : nx->src[0];
                if (oth->type == GGML_TYPE_F32 && ggml_are_same_shape(oth, dst) && ggml_are_same_shape(nx, dst) &&
                    ggml_is_contiguous(oth) && ggml_is_contiguous(nx) && ggml_is_contiguous(dst)) {
                    add = nx;
                }
            }
        }
    }
    if (add && ctx->mem_ranges && !ggml_mem_ranges_check(ctx->mem_ranges, add)) {
        ggml_metal_op_concurrency_reset(ctx);
    }

    auto pipeline = add ? ggml_metal_library_get_pipeline_moe_reduce_add(lib, (int32_t) experts->ne[1])
                        : ggml_metal_library_get_pipeline_moe_reduce    (lib, (int32_t) experts->ne[1]);

    const int nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int n_col_tiles = (args.ne00 + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(experts), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),     3);
    if (add) {
        const ggml_tensor * oth = add->src[0] == dst ? add->src[1] : add->src[0];
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(oth), 4);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(add), 5);
    }
"""
s = s.replace(old, new)
old = """    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: MOE_REDUCE\\n", __func__);
    }

    return n_fuse;
}"""
old = old.replace("\\\\n", chr(92) + "n")
assert s.count(old) == 1, s.count(old)
s = s.replace(old, old.replace("    return n_fuse;\n}", "    return add ? n_fuse + 1 : n_fuse;\n}"))
open(p, "w").write(s)
print("moeadd ok")
