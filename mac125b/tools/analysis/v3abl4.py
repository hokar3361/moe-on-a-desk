# timing-only ablations of kernel_moe_v3_q2_0 (argv[2] = mode)
#   1: no activation loads   2: no weight loads   3: neither (pure structure: grouping, loop, reduction, stores)
#   4: return right after moe2_group (launch + grouping only)
import sys
p = sys.argv[1] + "/ggml/src/ggml-metal/kernels/mul_mv.metal"
mode = int(sys.argv[2])
s = open(p).read()
old = "                    const float4 a = y4[0], b = y4[1], e = y4[2], f = y4[3];\n"
assert s.count(old) == 1
if mode in (1, 3):
    s = s.replace(old, "                    const float4 a = float4(c*1e-3f, 1.0f, 2.0f, 3.0f) + yb[t][0], b = a + 1.0f, e = a + 2.0f, f = a + 3.0f; (void) y4;\n")
old = """                    device const block_q2_0 * xb = xr[r] + c/4;
                    device const uint16_t * q16 = (device const uint16_t *) xb->qs + 2*(c%4);
                    wc[r] = (uint) q16[0] | ((uint) q16[1] << 16);
                    dc[r] = (float) xb->d;
"""
assert s.count(old) == 1
if mode in (2, 3):
    s = s.replace(old, """                    wc[r] = wn[r] + c*0x01010101u + r; dc[r] = dn[r];
""")
old = """    if (!moe2_group(args, ids, pz, i02, tiisg, grp, ng)) {
        return;
    }

    const short lr = tiisg/8;   // row within the step"""
assert s.count(old) == 1, s.count(old)
if mode == 4:
    s = s.replace(old, old.replace("        return;\n    }\n", "        return;\n    }\n    if (ng < 100) { return; }\n"))
open(p, "w").write(s)
print("v3abl4 ok", mode)
