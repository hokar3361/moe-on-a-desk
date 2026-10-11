# Metal: bitonic top-k block sort with the (value, index) pair in registers. Same network, same comparisons and swaps as
# kernel_argsort_f32_i32 (so the output is bit-identical); partners within a SIMD group are exchanged with simd_shuffle_xor
# (no barrier), farther partners through threadgroup memory. The original re-reads both values from device memory through
# the index on every compare and needs a barrier on all 55 steps of a 1024-wide sort (QSA indexer top-k: ~160 us per call).
# GGML_METAL_TOPK_REG=0 restores the original kernel.
import sys
root = sys.argv[1] + "/ggml/src/ggml-metal/"

p = root + "kernels/argsort.metal"
s = open(p).read()
anchor = "typedef void (argsort_merge_t)("
assert s.count(anchor) == 1
s = s.replace(anchor, """template<ggml_sort_order order>
kernel void kernel_argsort_reg_f32_i32(
        constant   ggml_metal_kargs_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);

    int   idx = i00 + col;
    float val = idx < args.ne00 ? src0_row[idx] : 0.0f;

    for (int k = 2; k <= (int) ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            int   pidx;
            float pval;
            if (j >= 32) {
                shmem_i32[col] = idx;
                shmem_f32[col] = val;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                pidx = shmem_i32[ixj];
                pval = shmem_f32[ixj];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                pidx = simd_shuffle_xor(idx, (ushort) j);
                pval = simd_shuffle_xor(val, (ushort) j);
            }
            // the original decides on the lower element of the pair; both threads evaluate the same decision
            const bool  is_lo = ixj > col;
            const int   lo_i  = is_lo ? idx  : pidx;
            const int   hi_i  = is_lo ? pidx : idx;
            const float lo_v  = is_lo ? val  : pval;
            const float hi_v  = is_lo ? pval : val;
            bool sw;
            if ((col & k) == 0) {
                sw = lo_i >= args.ne00 ||
                    (hi_i <  args.ne00 && (order == GGML_SORT_ORDER_ASC ? lo_v > hi_v : lo_v < hi_v));
            } else {
                sw = hi_i >= args.ne00 ||
                    (lo_i <  args.ne00 && (order == GGML_SORT_ORDER_ASC ? lo_v < hi_v : lo_v > hi_v));
            }
            if (sw) {
                idx = pidx;
                val = pval;
            }
        }
    }

    const int64_t i0 = ib*args.top_k;

    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = idx;
    }
}

template [[host_name("kernel_argsort_reg_f32_i32_asc")]]  kernel argsort_t kernel_argsort_reg_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_reg_f32_i32_desc")]] kernel argsort_t kernel_argsort_reg_f32_i32<GGML_SORT_ORDER_DESC>;

""" + anchor)
open(p, "w").write(s)

p = root + "ggml-metal-device.cpp"
s = open(p).read()
old = """    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_radix("""
assert s.count(old) == 1
s = s.replace(old, """    // GGML_METAL_TOPK_REG=0: the original kernel (the register version gives the same output)
    static const bool reg = !getenv("GGML_METAL_TOPK_REG") || atoi(getenv("GGML_METAL_TOPK_REG")) != 0;
    snprintf(base, 256, reg ? "kernel_argsort_reg_%s_%s_%s" : "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_radix(""")
open(p, "w").write(s)

p = root + "ggml-metal-ops.cpp"
s = open(p).read()
old = """    // blocks per row
    const int npr = (ne00 + nth - 1)/nth;

    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += sizeof(int32_t)*ggml_nelements(op->src[0]);"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, """    // blocks per row
    const int npr = (ne00 + nth - 1)/nth;

    // indices + values (the register kernel exchanges both through threadgroup memory)
    const size_t smem = GGML_PAD(2*nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += sizeof(int32_t)*ggml_nelements(op->src[0]);""")
open(p, "w").write(s)
print("topkreg ok")
