# RMS_NORM + MUL + MUL (gated norm: norm(x)*w*g) as one dispatch: ((x*scale)*f0)*f1, products only (no contraction)
import sys
R = sys.argv[1] + "/ggml/src/ggml-metal/"

p = R + "kernels/norm.metal"
s = open(p).read()
if "kernel_rms_norm_mul_mul_f32" not in s:
    old = """        if (F == 3) {
            y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_rms_norm_fuse_impl<float4, 1>) kernel_rms_norm_fuse_t;"""
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, """        if (F == 3) {
            y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];
        }
        if (F == 4) {
            y[i00] = ((x[i00]*scale)*f0[i00])*f1[i00];
        }
    }
}

typedef decltype(kernel_rms_norm_fuse_impl<float4, 1>) kernel_rms_norm_fuse_t;""")
    old = """template [[host_name("kernel_rms_norm_mul_add_f32_4")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 3>;"""
    assert s.count(old) == 1
    s = s.replace(old, old + """
template [[host_name("kernel_rms_norm_mul_mul_f32")]]   kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 4>;
template [[host_name("kernel_rms_norm_mul_mul_f32_4")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 4>;""")
    open(p, "w").write(s)

p = R + "ggml-metal-fusion.h"
s = open(p).read()
if "GGML_METAL_FUSION_NORM_MUL_MUL" not in s:
    old = "    GGML_METAL_FUSION_NORM_MUL_ADD, // NORM/RMS_NORM + MUL + ADD"
    assert s.count(old) == 1
    s = s.replace(old, old + "\n    GGML_METAL_FUSION_NORM_MUL_MUL, // RMS_NORM + MUL + MUL (gated norm)")
    open(p, "w").write(s)

p = R + "ggml-metal-fusion.cpp"
s = open(p).read()
if "ops_rms_norm_mul_mul" not in s:
    old = "static const std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };"
    assert s.count(old) == 1
    s = s.replace(old, old + "\nstatic const std::vector<ggml_op> ops_rms_norm_mul_mul = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL };")
    old = "    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     false, ggml_metal_fusion_check_norm },"
    assert s.count(old) == 1
    s = s.replace(old, old + "\n    { GGML_METAL_FUSION_NORM_MUL_MUL,   ops_rms_norm_mul_mul,       {},     false, ggml_metal_fusion_check_norm },")
    open(p, "w").write(s)

p = R + "ggml-metal-ops.cpp"
s = open(p).read()
if "GGML_METAL_FUSION_NORM_MUL_MUL" not in s:
    old = "        if (fusion && (ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL || ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL_ADD)) {"
    assert s.count(old) == 1
    s = s.replace(old, """        norm_mul_mul = fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL_MUL;
        if (fusion && (ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL || ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL_ADD || norm_mul_mul)) {""")
    old = """    auto pipeline = fused_norm_scale ?
        ggml_metal_library_get_pipeline_norm_scale(lib, op) :
        ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);"""
    assert s.count(old) == 1
    s = s.replace(old, """    ggml_metal_pipeline_with_params pipeline_mm = {};
    if (norm_mul_mul) {
        const char * nm = op->ne[0] % 4 == 0 ? "kernel_rms_norm_mul_mul_f32_4" : "kernel_rms_norm_mul_mul_f32";
        pipeline_mm = ggml_metal_library_get_pipeline(lib, nm);
        if (!pipeline_mm.pipeline) {
            ggml_metal_cv_t cv = ggml_metal_cv_init();
            ggml_metal_cv_set_bool(cv, false, FC_NORM + 0);
            pipeline_mm = ggml_metal_library_compile_pipeline(lib, nm, nm, cv);
            ggml_metal_cv_free(cv);
        }
        pipeline_mm.smem = 32*sizeof(float);
    }
    auto pipeline = norm_mul_mul ? pipeline_mm : fused_norm_scale ?
        ggml_metal_library_get_pipeline_norm_scale(lib, op) :
        ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);""")
    old = """    int n_fuse = 1;
    bool fused_norm_scale = false;
"""
    assert s.count(old) == 1
    s = s.replace(old, old + "    bool norm_mul_mul = false;\n")
    open(p, "w").write(s)
print("normmm ok")
