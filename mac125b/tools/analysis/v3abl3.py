# timing-only ablation of kernel_moe_v3_q2_0: MOE3_NOACT=1 replaces the activation loads by register values
# (same arithmetic, no device reads of src1) -> how much of the kernel time is activation traffic
import sys
p = sys.argv[1] + "/ggml/src/ggml-metal/kernels/mul_mv.metal"
s = open(p).read()
old = "                    const float4 a = y4[0], b = y4[1], e = y4[2], f = y4[3];\n"
assert s.count(old) == 1
new = """#if MOE3_NOACT
                    const float4 a = float4(c*1e-3f, 1.0f, 2.0f, 3.0f) + yb[t][0], b = a + 1.0f, e = a + 2.0f, f = a + 3.0f; (void) y4;
#else
                    const float4 a = y4[0], b = y4[1], e = y4[2], f = y4[3];
#endif
"""
s = s.replace(old, new)
s = s.replace("#define MOE3_PF ", "#define MOE3_NOACT " + sys.argv[2] + "\n#define MOE3_PF ", 1)
open(p, "w").write(s)
print("v3abl3 ok")
