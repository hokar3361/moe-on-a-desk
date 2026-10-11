# HC_POST (gated) + RMS_NORM + MUL(w) in one dispatch: one threadgroup per (stream, token), the same thread count and
# reduction order as kernel_rms_norm_mul_f32_4 (one float4 per thread, simd_sum, then across simdgroups) -> bit-identical.
# GGML_METAL_FUSE_HC_POST_NORM=0 disables
import sys
R = sys.argv[1] + "/ggml/src/ggml-metal/"

p = R + "kernels/misc.metal"
s = open(p).read()
if "kernel_hc_post_norm_f32" not in s:
    s += """
// gated hyper-connection combine followed by the grouped RMS norm and its gamma of the next mixer
//   res[s] = x*post_s + residual[s];  xn[s] = (res[s]*rsqrt(mean(res[s]^2) + eps))*w[s]
// threadgroup (stream s, token t), args.n_embd/4 threads, one float4 each
kernel void kernel_hc_post_norm_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        constant float & eps,
        device const char * x,
        device const char * residual,
        device const char * post,
        device       char * dst,
        device const char * w,
        device       char * dst_n,
        constant uint64_t & nb_w1,
        constant uint64_t & nb_n1,
        constant uint64_t & nb_n2,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int is = tgpig.x;   // stream
    const int it = tgpig.y;   // token

    float p = *(device const float *) (post + is*args.nb_p0 + it*args.nb_p1);
    if (args.post_gate) {
        p = args.post_s_in*p + args.post_b_in;
        p = 1 / (1 + exp(-p));
        p = args.post_s_out*p + args.post_b_out;
    }

    const int i4 = tpitg.x;
    device const float4 * xv = (device const float4 *) (x + it*args.nb_x1);
    device const float4 * rv = (device const float4 *) (residual + is*args.nb_r1 + it*args.nb_r2);
    device       float4 * dv = (device       float4 *) (dst + is*args.nb_d1 + it*args.nb_d2);

    float4 v = 0.0f;
    float sumf = 0.0f;
    if (i4 < args.n_embd/4) {
        const float4 a = xv[i4], b = rv[i4];
        v = float4(a[0]*p + b[0], a[1]*p + b[1], a[2]*p + b[2], a[3]*p + b[3]);
        dv[i4] = v;
        sumf = dot(v, v);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.n_embd;
    const float scale = 1.0f/sqrt(mean + eps);

    if (i4 < args.n_embd/4) {
        device const float4 * wv = (device const float4 *) (w + is*nb_w1);
        device       float4 * nv = (device       float4 *) (dst_n + is*nb_n1 + it*nb_n2);
        nv[i4] = (v*scale)*wv[i4];
    }
}
"""
    open(p, "w").write(s)

p = R + "ggml-metal-ops.cpp"
s = open(p).read()
old = """    ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),        1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(residual), 2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(inject),   3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),       4);

    const int n_tiles = (args.n_embd + 31)/32;
    const int nsg = std::min(4, n_tiles);
    ggml_metal_encoder_dispatch_threadgroups(
            enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);

    return n_fuse;
}"""
assert s.count(old) == 1, s.count(old)
new = """    // the grouped RMS norm (+ gamma) of the next mixer reads this output: fold it in (same threads, same reduction order)
    {
        static const bool en = getenv("GGML_METAL_FUSE_HC_POST_NORM") == nullptr || atoi(getenv("GGML_METAL_FUSE_HC_POST_NORM")) != 0;
        if (en && ctx->use_fusion() && idx + n_fuse + 1 < ctx->n_nodes()) {
            ggml_tensor * nrm = ctx->node(idx + n_fuse);
            ggml_tensor * mul = ctx->node(idx + n_fuse + 1);
            const ggml_tensor * wn = mul->src[1];
            if (nrm->op == GGML_OP_RMS_NORM && nrm->src[0] == op && mul->op == GGML_OP_MUL && mul->src[0] == nrm &&
                op->ne[0] % 4 == 0 && op->ne[0]/4 <= 1024 && op->ne[1] == 4 &&
                ggml_are_same_shape(nrm, op) && ggml_are_same_shape(mul, op) && wn->type == GGML_TYPE_F32 &&
                wn->ne[0] == op->ne[0] && wn->ne[1] == 4 && (wn->ne[2] == 1) && wn->nb[0] == 4 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(mul) && x->nb[0] == 4 && residual->nb[0] == 4 &&
                x->nb[1] % 16 == 0 && residual->nb[1] % 16 == 0 && residual->nb[2] % 16 == 0 && wn->nb[1] % 16 == 0) {
                if (ctx->mem_ranges && (!ggml_mem_ranges_check(ctx->mem_ranges, nrm) || !ggml_mem_ranges_check(ctx->mem_ranges, mul))) {
                    ggml_metal_op_concurrency_reset(ctx);
                }
                float eps;
                memcpy(&eps, nrm->op_params, sizeof(float));
                const uint64_t nb_w1 = wn->nb[1], nb_n1 = mul->nb[1], nb_n2 = mul->nb[2];
                static ggml_metal_pipeline_with_params pl = {};
                if (!pl.pipeline) {
                    pl = ggml_metal_library_get_pipeline(ctx->lib, "kernel_hc_post_norm_f32");
                    if (!pl.pipeline) {
                        pl = ggml_metal_library_compile_pipeline(ctx->lib, "kernel_hc_post_norm_f32", "kernel_hc_post_norm_f32", nullptr);
                    }
                }
                ggml_metal_encoder_set_pipeline(enc, pl);
                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_bytes (enc, &eps, sizeof(eps), 1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),        2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(residual), 3);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(inject),   4);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),       5);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(wn),       6);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(mul),      7);
                ggml_metal_encoder_set_bytes (enc, (void *) &nb_w1, sizeof(nb_w1), 8);
                ggml_metal_encoder_set_bytes (enc, (void *) &nb_n1, sizeof(nb_n1), 9);
                ggml_metal_encoder_set_bytes (enc, (void *) &nb_n2, sizeof(nb_n2), 10);
                ggml_metal_encoder_set_threadgroup_memory_size(enc, 32*sizeof(float), 0);
                // the norm kernel's thread count: next power of two >= n/4, capped at n/4 rounded up to the simd width
                int nth = 32;
                const int n4 = (int) (op->ne[0]/4);
                while (nth < n4 && nth < 1024) nth *= 2;
                nth = std::min(nth, (n4 + 31)/32*32);
                ggml_metal_encoder_dispatch_threadgroups(enc, 4, args.n_tokens, 1, nth, 1, 1);
                return n_fuse + 2;
            }
        }
    }

""" + old.replace("    return n_fuse;\n}", "    return n_fuse;\n}").split("\n\n    const int n_tiles")[0] + """

    const int n_tiles = (args.n_embd + 31)/32;
    const int nsg = std::min(4, n_tiles);
    ggml_metal_encoder_dispatch_threadgroups(
            enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);

    return n_fuse;
}"""
# the fused branch must come before the regular buffer bindings: place it right before them
s = s.replace(old, new)
open(p, "w").write(s)
print("hcpostnorm ok")
