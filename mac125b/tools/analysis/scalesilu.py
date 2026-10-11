# SCALE followed by SILU of its result -> one dispatch (both outputs written, same arithmetic as the two kernels)
# GGML_METAL_FUSE_SCALE_SILU=0 disables
import sys
R = sys.argv[1] + "/ggml/src/ggml-metal/"
p = R + "kernels/unary.metal"
s = open(p).read()
if "kernel_scale_silu_f32" not in s:
    s += """
// fused SCALE + SILU over contiguous f32: t = scale*x + bias (written), y = t/(1 + exp(-t)) (written)
kernel void kernel_scale_silu_f32(
        constant ggml_metal_kargs_unary & args,
        device const float * src0,
        device       float * dst_t,
        device       float * dst_y,
        uint tid[[thread_position_in_grid]]) {
    if ((int) tid >= args.ne0) {
        return;
    }
    const float x = src0[tid];
    const float t = (float) (args.scale * x + args.bias);
    dst_t[tid] = t;
    dst_y[tid] = (float) (t / (1 + exp(-t)));
}
"""
    open(p, "w").write(s)

p = R + "ggml-metal-ops.cpp"
s = open(p).read()
old = """            return ggml_metal_op_hc_post_gate(ctx, idx, n);
        }
    }
"""
assert s.count(old) == 1, s.count(old)
new = old + """
    // SCALE + SILU of its result in one dispatch
    if (op->op == GGML_OP_SCALE && ctx->use_fusion() && idx + 1 < ctx->n_nodes()) {
        static const bool en = getenv("GGML_METAL_FUSE_SCALE_SILU") == nullptr || atoi(getenv("GGML_METAL_FUSE_SCALE_SILU")) != 0;
        ggml_tensor * nx = ctx->node(idx + 1);
        if (en && nx->op == GGML_OP_UNARY && ggml_get_unary_op(nx) == GGML_UNARY_OP_SILU && nx->src[0] == op &&
            op->type == GGML_TYPE_F32 && nx->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op) && ggml_is_contiguous(nx) &&
            ggml_nelements(op) == ggml_nelements(op->src[0]) && ggml_nelements(nx) == ggml_nelements(op)) {
            if (ctx->mem_ranges && !ggml_mem_ranges_check(ctx->mem_ranges, nx)) {
                ggml_metal_op_concurrency_reset(ctx);
            }
            ggml_metal_kargs_unary args = {};
            args.ne0   = (int32_t) ggml_nelements(op);
            args.scale = ggml_get_op_params_f32(op, 0);
            args.bias  = ggml_get_op_params_f32(op, 1);
            static ggml_metal_pipeline_with_params pl = {};
            if (!pl.pipeline) {
                pl = ggml_metal_library_get_pipeline(ctx->lib, "kernel_scale_silu_f32");
                if (!pl.pipeline) {
                    pl = ggml_metal_library_compile_pipeline(ctx->lib, "kernel_scale_silu_f32", "kernel_scale_silu_f32", nullptr);
                }
            }
            ggml_metal_encoder_set_pipeline(ctx->enc, pl);
            ggml_metal_encoder_set_bytes   (ctx->enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(op->src[0]), 1);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(op), 2);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(nx), 3);
            const int n = args.ne0;
            const int nth = std::min(256, n);
            ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (n + nth - 1)/nth, 1, 1, nth, 1, 1);
            return 2;
        }
    }

"""
s = s.replace(old, new)
open(p, "w").write(s)
print("scalesilu ok")
