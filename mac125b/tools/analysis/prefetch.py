# GGML_METAL_PREFETCH_KB=N: weight prefetch into the SLC during DRAM-idle stages.
# After every "light" node (not a matmul / attention / delta-net), a small read-only dispatch in the same stage touches
# the next N KB of the weights of the next dense MUL_MAT (up to GGML_METAL_PREFETCH_AHEAD_MB ahead), so that matmul
# starts with its first weights already in the SLC instead of ramping up DRAM after the barrier.
# GGML_METAL_PREFETCH_MOE=1: also prefetch (GGML_METAL_PREFETCH_MOE_KB) during MUL_MAT_ID stages (ALU-bound, DRAM idle).
# The dispatch reads only weights and writes nothing (a sink write that cannot happen), so results are unchanged.
import sys
R = sys.argv[1] + "/ggml/src/ggml-metal/"
BS = chr(92)

p = R + "kernels/misc.metal"
s = open(p).read()
if "kernel_prefetch_w" not in s:
    s += """
// read-only touch of a weight range (SLC prefetch); the sink write never happens in practice
kernel void kernel_prefetch_w(
        device const uint4 * src  [[buffer(0)]],
        device       uint  * sink [[buffer(1)]],
        constant     uint  & n    [[buffer(2)]],
        uint tid  [[thread_position_in_grid]],
        uint ntot [[threads_per_grid]]) {
    uint4 acc = uint4(0);
    for (uint i = tid; i < n; i += ntot) {
        acc ^= src[i];
    }
    if (acc.x == 0x9E3779B9u && acc.y == 0x7F4A7C15u && acc.z == 0x85EBCA6Bu && acc.w == 0xC2B2AE35u) {
        sink[tid & 255] = acc.x;
    }
}
"""
    open(p, "w").write(s)

p = R + "ggml-metal-ops.cpp"
s = open(p).read()
helper = """
// ---- SLC weight prefetch during DRAM-idle stages (GGML_METAL_PREFETCH_KB) ----
static bool ggml_metal_pf_heavy(const ggml_tensor * t) {
    return t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID || t->op == GGML_OP_FLASH_ATTN_EXT ||
           t->op == GGML_OP_GATED_DELTA_NET;
}

static void ggml_metal_op_prefetch(ggml_metal_op_t ctx, int idx_next, size_t kb) {
    static const size_t ahead = (size_t) (getenv("GGML_METAL_PREFETCH_AHEAD_MB") ? atof(getenv("GGML_METAL_PREFETCH_AHEAD_MB")) : 24.0)*1024*1024;
    static int    cur_j = -1;                   // graph index of the matmul being prefetched
    static size_t off   = 0;                    // bytes of its weights already prefetched
    static int                 last_idx = 1 << 30;
    if (idx_next < last_idx) { cur_j = -1; off = 0; }   // new graph
    last_idx = idx_next;

    size_t budget = kb*1024;
    size_t dist   = 0;                          // weight bytes between here and the prefetch position
    const int n = ctx->n_nodes();
    for (int j = idx_next; j < n && budget > 0 && dist < ahead; ++j) {
        const ggml_tensor * t = ctx->node(j);
        if (t->op != GGML_OP_MUL_MAT) {
            continue;
        }
        const ggml_tensor * w = t->src[0];
        if (!w->buffer || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS || !ggml_is_contiguous(w)) {
            continue;
        }
        const size_t nb = ggml_nbytes(w);
        size_t done = 0;
        if (j < cur_j) {
            dist += nb; continue;          // passed or already fully prefetched
        } else if (j == cur_j) {
            done = off;
        }
        if (done >= nb) { dist += nb; continue; }
        const size_t len = std::min(budget, nb - done) & ~(size_t) 15;
        if (len == 0) { break; }

        static ggml_metal_pipeline_with_params pl = {};
        if (!pl.pipeline) {
            pl = ggml_metal_library_get_pipeline(ctx->lib, "kernel_prefetch_w");
            if (!pl.pipeline) {
                pl = ggml_metal_library_compile_pipeline(ctx->lib, "kernel_prefetch_w", "kernel_prefetch_w", nullptr);
            }
        }
        ggml_metal_buffer_id bid = ggml_metal_get_buffer_id(w);
        bid.offs += done;
        ggml_metal_buffer_id sink = ggml_metal_device_get_sync_buffer(ctx->dev);
        sink.offs += 32*1024;
        const uint32_t n16 = (uint32_t) (len/16);
        const int nth = (int) std::min<uint32_t>(n16, 16384);
        ggml_metal_encoder_set_pipeline(ctx->enc, pl);
        ggml_metal_encoder_set_buffer  (ctx->enc, bid, 0);
        ggml_metal_encoder_set_buffer  (ctx->enc, sink, 1);
        ggml_metal_encoder_set_bytes   (ctx->enc, (void *) &n16, sizeof(n16), 2);
        ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (nth + 255)/256, 1, 1, 256, 1, 1);

        cur_j = j; off = done + len;
        budget -= len;
        dist += nb;
    }
}

"""
anchor = "static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor)
old = """    // update the mem ranges in the encoding context
    for (int i = 0; i < n_fuse; ++i) {
        if (!ggml_metal_op_concurrency_add(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }

    return n_fuse;
}"""
assert s.count(old) == 1
new = """    // update the mem ranges in the encoding context
    for (int i = 0; i < n_fuse; ++i) {
        if (!ggml_metal_op_concurrency_add(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }

    {
        static const size_t pf_kb     = getenv("GGML_METAL_PREFETCH_KB") ? (size_t) atoi(getenv("GGML_METAL_PREFETCH_KB")) : 0;
        static const size_t pf_moe_kb = getenv("GGML_METAL_PREFETCH_MOE_KB") ? (size_t) atoi(getenv("GGML_METAL_PREFETCH_MOE_KB")) : 0;
        const ggml_tensor * last = ctx->node(idx + n_fuse - 1);
        if (pf_kb > 0 && !ggml_metal_pf_heavy(last)) {
            ggml_metal_op_prefetch(ctx, idx + n_fuse, pf_kb);
        } else if (pf_moe_kb > 0 && last->op == GGML_OP_MUL_MAT_ID) {
            ggml_metal_op_prefetch(ctx, idx + n_fuse, pf_moe_kb);
        }
    }

    return n_fuse;
}"""
s = s.replace(old, new)
open(p, "w").write(s)
print("prefetch ok")
