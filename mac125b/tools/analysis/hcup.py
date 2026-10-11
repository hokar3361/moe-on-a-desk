# HC mix tail in one dispatch: SCALE -> SILU -> bf16 MUL_MAT (up, 4*n_embd rows) -> DSV4_HC_PRE (gated)
#  - the few-row MMA kernel with the SILU(scale*x + bias) applied to src1 as it is loaded (same expressions as the
#    SCALE and SILU kernels), src0 rows read in the order 4*e + stream (row remap only, same weights and MMA),
#    and the gated HC_PRE computed from the reduced tile sums (same expressions and order as kernel_dsv4_hc_pre_gated_f32)
#  - bit-identical to the unfused path when the unfused path also runs the few-row MMA (B >= 2)
# GGML_METAL_FUSE_HC_UP=0 disables
import sys
R = sys.argv[1] + "/ggml/src/ggml-metal/"

p = R + "kernels/mul_mv_mma.metal"
s = open(p).read()
if "kernel_mul_mv_mma_bf16_hc" not in s:
    s += """
#if defined(GGML_METAL_HAS_BF16)
struct mma_hc_args {
    float    s_in;      // SCALE scale
    float    b_in;      // SCALE bias
    float    pre_scale; // HC_PRE scale
    int32_t  n_embd;
    uint64_t nb_x1;     // xn: stream stride
    uint64_t nb_x2;     // xn: token stride
    uint64_t nb_d1;     // mixed: token stride
};

inline float4 mma_hc_silu(float4 v, float s, float b) {
    float4 r;
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        const float t = (float) (s*v[i] + b);
        r[i] = (float) (t / (1 + exp(-t)));
    }
    return r;
}

template<short NT, short RT>
kernel void kernel_mul_mv_mma_bf16_hc(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        constant mma_hc_args & ha,
        device const char * xn,
        device       char * mixed,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_mma_nsg;

    const mul_mv_mma_tile tile = mul_mv_mma_tile_init<NT, RT>(args, tgpig, tiisg);

    // logical row 4*e + s reads the weights of stream s, element e
    const int nq = args.ne01/4;
    device const bfloat4x4 * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        const int r = min(tile.i01 + 8*t + tile.fm, args.ne01 - 1);
        const int phys = (r % 4)*nq + r/4;
        x[t] = (device const bfloat4x4 *) (src0 + tile.offset0 + (uint64_t) phys*args.nb01);
    }

    device const float4 * y[RT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            y[rt][e] = (device const float4 *) mul_mv_mma_src1_row(tile, args, src1, rt, e) + 4*(tile.fm/2) + 2*(tile.fm%2);
        }
    }

    simdgroup_float8x8 mc[RT][NT];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            mc[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    const int nch = args.ne00/64;

    for (int g = sgitg; g < nch; g += NSG) {
        simdgroup_float8x8 mb[RT][8];
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            const float4 a0 = mma_hc_silu(y[rt][0][16*g + 0], ha.s_in, ha.b_in);
            const float4 a1 = mma_hc_silu(y[rt][0][16*g + 1], ha.s_in, ha.b_in);
            const float4 b0 = mma_hc_silu(y[rt][1][16*g + 0], ha.s_in, ha.b_in);
            const float4 b1 = mma_hc_silu(y[rt][1][16*g + 1], ha.s_in, ha.b_in);
            FOR_UNROLL (short s = 0; s < 4; ++s) {
                mb[rt][s    ].thread_elements()[0] = a0[s];
                mb[rt][s    ].thread_elements()[1] = b0[s];
                mb[rt][s + 4].thread_elements()[0] = a1[s];
                mb[rt][s + 4].thread_elements()[1] = b1[s];
            }
        }

        const int ci = 4*g + tile.fn/2;

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            float4x4 w;
            dequantize_bf16(x[t] + ci, 0, w);

            FOR_UNROLL (short s = 0; s < 8; ++s) {
                simdgroup_float8x8 ma;
                ma.thread_elements()[0] = w[s/4    ][s%4];
                ma.thread_elements()[1] = w[s/4 + 2][s%4];

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    simdgroup_multiply_accumulate(mc[rt][t], ma, mb[rt][s], mc[rt][t]);
                }
            }
        }
    }

    // reduce the K slices exactly as mul_mv_mma_store, into a [8*NT rows][8*RT columns] tile of gate values
    threadgroup float * red = (threadgroup float *) shmem;
    threadgroup float * gb  = red + NSG*RT*NT*64;

    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            red[((sgitg*RT + rt)*NT + t)*64 + 2*tiisg + 0] = mc[rt][t].thread_elements()[0];
            red[((sgitg*RT + rt)*NT + t)*64 + 2*tiisg + 1] = mc[rt][t].thread_elements()[1];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short idx = sgitg*32 + tiisg; idx < RT*NT*64; idx += NSG*32) {
        float sum = 0.0f;
        for (short sg = 0; sg < NSG; ++sg) {
            sum += red[sg*(RT*NT*64) + idx];
        }

        const short rt = idx/(NT*64);
        const short t  = (idx/64) % NT;
        const short l  = (idx % 64)/2;
        const short e  = idx % 2;

        const short r0l = 8*t  + mul_mv_mma_lane_fm(l);
        const short r1l = 8*rt + mul_mv_mma_lane_fn(l) + e;

        gb[r0l*(8*RT) + r1l] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // gated HC_PRE: mixed[e] = scale * sum_s xn[s][e]*sigmoid(gate[s][e]), streams in order
    for (short q = sgitg*32 + tiisg; q < 2*NT*8*RT; q += NSG*32) {
        const short el = q/(8*RT);
        const short c  = q%(8*RT);
        const int   e  = tile.i01/4 + el;
        const int   r1 = tile.i11 + c;
        if (e < ha.n_embd && r1 < args.ne11) {
            float result = 0.0f;
            FOR_UNROLL (short ih = 0; ih < 4; ++ih) {
                const float g  = 1.0f/(1.0f + exp(-gb[(4*el + ih)*(8*RT) + c]));
                const float xv = *(device const float *) (xn + (uint64_t) e*4 + ih*ha.nb_x1 + (uint64_t) r1*ha.nb_x2);
                result = fma(xv, g, result);
            }
            *(device float *) (mixed + (uint64_t) e*4 + (uint64_t) r1*ha.nb_d1) = ha.pre_scale*result;
        }
    }
}

typedef decltype(kernel_mul_mv_mma_bf16_hc<1, 1>) kernel_mul_mv_mma_bf16_hc_t;
template [[host_name("kernel_mul_mv_mma_bf16_hc_nt1_rt1")]] kernel kernel_mul_mv_mma_bf16_hc_t kernel_mul_mv_mma_bf16_hc<1, 1>;
template [[host_name("kernel_mul_mv_mma_bf16_hc_nt2_rt1")]] kernel kernel_mul_mv_mma_bf16_hc_t kernel_mul_mv_mma_bf16_hc<2, 1>;
template [[host_name("kernel_mul_mv_mma_bf16_hc_nt4_rt1")]] kernel kernel_mul_mv_mma_bf16_hc_t kernel_mul_mv_mma_bf16_hc<4, 1>;
template [[host_name("kernel_mul_mv_mma_bf16_hc_nt1_rt2")]] kernel kernel_mul_mv_mma_bf16_hc_t kernel_mul_mv_mma_bf16_hc<1, 2>;
template [[host_name("kernel_mul_mv_mma_bf16_hc_nt2_rt2")]] kernel kernel_mul_mv_mma_bf16_hc_t kernel_mul_mv_mma_bf16_hc<2, 2>;
template [[host_name("kernel_mul_mv_mma_bf16_hc_nt4_rt2")]] kernel kernel_mul_mv_mma_bf16_hc_t kernel_mul_mv_mma_bf16_hc<4, 2>;
#endif
"""
    open(p, "w").write(s)

p = R + "ggml-metal-device.cpp"
s = open(p).read()
if "get_pipeline_mul_mv_mma_hc" not in s:
    anchor = "// the pipeline for the tiling, with fewer simdgroups when the device cannot run that many threads"
    assert s.count(anchor) == 1
    s = s.replace(anchor, """// the fused HC mix tail (SCALE -> SILU -> bf16 up -> gated HC_PRE), with the tiling of the plain mat-mul
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_hc(ggml_metal_library_t lib, const ggml_tensor * op) {
    ggml_metal_mma_tiling tiling = ggml_metal_op_mul_mat_mma_tiling(op);

    ggml_metal_pipeline_with_params res = {};
    while (true) {
        char base[256];
        char name[256];
        snprintf(base, 256, "kernel_mul_mv_mma_bf16_hc_nt%d_rt%d", tiling.nt, tiling.rt);
        snprintf(name, 256, "%s_nsg=%d", base, tiling.nsg);

        res = ggml_metal_library_get_pipeline(lib, name);
        if (!res.pipeline) {
            ggml_metal_cv_t cv = ggml_metal_cv_init();
            ggml_metal_cv_set_int32(cv, 0,                 FC_MUL_MV_MMA + 4);
            ggml_metal_cv_set_int16(cv, tiling.nsg,        FC_MUL_MV_MMA + 0);
            ggml_metal_cv_set_int16(cv, 1,                 FC_MUL_MV_MMA + 1);
            ggml_metal_cv_set_int16(cv, 1,                 FC_MUL_MV_MMA + 2);
            ggml_metal_cv_set_int16(cv, 1,                 FC_MUL_MV_MMA + 3);
            ggml_metal_cv_set_bool (cv, false,             FC_MUL_MV_MMA + 5);
            res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
            ggml_metal_cv_free(cv);
        }
        if (tiling.nsg > 1 && res.pipeline && ggml_metal_pipeline_max_theads_per_threadgroup(res) < tiling.nsg*32) {
            tiling.nsg /= 2;
            continue;
        }
        break;
    }

    res.nsg  = tiling.nsg;
    res.smem = ggml_metal_mul_mv_mma_smem(tiling.nsg + 1, tiling.nt, tiling.rt);
    res.nr0  = GGML_METAL_MMA_TILE*tiling.nt;
    res.nr1  = GGML_METAL_MMA_TILE*tiling.rt;

    return res;
}

""" + anchor)
    open(p, "w").write(s)

p = R + "ggml-metal-device.h"
s = open(p).read()
if "get_pipeline_mul_mv_mma_hc" not in s:
    anchor = "struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, const struct ggml_tensor * op, bool add);"
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + "\nstruct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_hc     (ggml_metal_library_t lib, const struct ggml_tensor * op);")
    open(p, "w").write(s)

p = R + "ggml-metal-ops.cpp"
s = open(p).read()
if "GGML_METAL_FUSE_HC_UP" not in s:
    anchor = """    // SCALE + SILU of its result in one dispatch
    if (op->op == GGML_OP_SCALE && ctx->use_fusion() && idx + 1 < ctx->n_nodes()) {"""
    assert s.count(anchor) == 1
    s = s.replace(anchor, """    // SCALE -> SILU -> bf16 MUL_MAT (4*n_embd rows) -> gated DSV4_HC_PRE in one dispatch (the HC mix tail)
    if (op->op == GGML_OP_SCALE && ctx->use_fusion() && idx + 3 < ctx->n_nodes()) {
        static const bool en = getenv("GGML_METAL_FUSE_HC_UP") == nullptr || atoi(getenv("GGML_METAL_FUSE_HC_UP")) != 0;
        ggml_tensor * si = ctx->node(idx + 1);
        ggml_tensor * mm = ctx->node(idx + 2);
        int jp = idx + 3;
        while (jp < ctx->n_nodes() && ggml_op_is_empty(ctx->node(jp)->op)) {
            jp++;
        }
        ggml_tensor * pre = jp < ctx->n_nodes() ? ctx->node(jp) : nullptr;
        const ggml_tensor * gate = pre ? pre->src[1] : nullptr;
        const ggml_tensor * xn   = pre ? pre->src[0] : nullptr;
        if (en && si->op == GGML_OP_UNARY && ggml_get_unary_op(si) == GGML_UNARY_OP_SILU && si->src[0] == op &&
            mm->op == GGML_OP_MUL_MAT && mm->src[1] == si && mm->src[0]->type == GGML_TYPE_BF16 && mm->type == GGML_TYPE_F32 &&
            pre && pre->op == GGML_OP_DSV4_HC_PRE && ggml_get_op_params_i32(pre, 1) == 1 && gate &&
            (gate == mm || gate->view_src == mm) && gate->view_offs == 0 && xn->ne[1] == 4 &&
            mm->src[0]->ne[1] == 4*xn->ne[0] && gate->ne[0] == xn->ne[0] && ggml_is_contiguous(gate) &&
            mm->src[0]->ne[0] % 64 == 0 && mm->src[0]->ne[2] == 1 && mm->src[0]->ne[3] == 1 &&
            si->ne[1] >= 2 && si->ne[1] <= 16 && si->ne[2] == 1 && si->ne[3] == 1 &&
            ggml_is_contiguous(op) && ggml_is_contiguous(si) && ggml_is_contiguous(op->src[0]) && ggml_nelements(op) == ggml_nelements(op->src[0]) &&
            xn->nb[0] == 4 && pre->nb[0] == 4 && op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32) {
            // the fused kernel reads the SCALE input and writes only the HC_PRE output: check that nothing in flight conflicts
            if (ctx->mem_ranges && (!ggml_mem_ranges_check(ctx->mem_ranges, op) || !ggml_mem_ranges_check(ctx->mem_ranges, pre))) {
                ggml_metal_op_concurrency_reset(ctx);
            }
            // the mat-mul reads the SCALE input in place of its src1: same row layout as the SILU output
            const ggml_tensor * w = mm->src[0];
            const ggml_tensor * in = op->src[0];
            ggml_metal_kargs_mul_mv_ext args = {};
            args.ne00 = (int32_t) w->ne[0];  args.ne01 = (int32_t) w->ne[1];  args.ne02 = 1;
            args.nb00 = w->nb[0]; args.nb01 = w->nb[1]; args.nb02 = w->nb[2]; args.nb03 = w->nb[3];
            args.ne10 = (int32_t) si->ne[0]; args.ne11 = (int32_t) si->ne[1]; args.ne12 = 1;
            args.nb10 = in->nb[0]; args.nb11 = in->nb[1]; args.nb12 = in->nb[2]; args.nb13 = in->nb[3];
            args.ne0  = (int32_t) mm->ne[0]; args.ne1 = (int32_t) mm->ne[1];
            args.r2 = 1; args.r3 = 1;
            struct {
                float    s_in;
                float    b_in;
                float    pre_scale;
                int32_t  n_embd;
                uint64_t nb_x1;
                uint64_t nb_x2;
                uint64_t nb_d1;
            } ha = {
                ggml_get_op_params_f32(op, 0), ggml_get_op_params_f32(op, 1), ggml_get_op_params_f32(pre, 0),
                (int32_t) xn->ne[0], xn->nb[1], xn->nb[2], pre->nb[1],
            };
            auto pipeline = ggml_metal_library_get_pipeline_mul_mv_mma_hc(ctx->lib, mm);
            ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
            ggml_metal_encoder_set_bytes   (ctx->enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(w),   1);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(in),  2);
            ggml_metal_encoder_set_bytes   (ctx->enc, &ha, sizeof(ha), 3);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(xn),  4);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(pre), 5);
            ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, pipeline.smem, 0);
            ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (args.ne01 + pipeline.nr0 - 1)/pipeline.nr0,
                    (args.ne11 + pipeline.nr1 - 1)/pipeline.nr1, 1, 32, pipeline.nsg, 1);
            return jp - idx + 1;
        }
    }

""" + anchor)
    open(p, "w").write(s)
print("hcup ok")
