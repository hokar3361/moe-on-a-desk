# Metal mul_mv_id for Q5_K rows of at most 2 super-blocks (the MoE down projections of Qwen3.6-35B-A3B, K = 512):
# the plain layout leaves the lanes with tiisg%4 >= 2 idle (their block index ix >= nb). With FC_mul_mv_split set by the
# host, those lanes compute a second group of nr0 rows with the same per-lane arithmetic (ix - 2); before the reduction
# their partials are moved to the plain layout's lane positions (simd_shuffle_xor 2), so every row is reduced from the
# same values in the same lanes as before: outputs bit-identical. GGML_METAL_Q5K_DUAL=0 restores the plain dispatch.
import sys
root = sys.argv[1]

def sub_in_func(s, start_marker, end_marker, pairs):
    a = s.index(start_marker); b = s.index(end_marker, a)
    body = s[a:b]
    for old, new in pairs:
        assert body.count(old) == 1, (old[:60], body.count(old))
        body = body.replace(old, new)
    return s[:a] + body + s[b:]

p = root + "/ggml/src/ggml-metal/kernels/mul_mv.metal"
s = open(p, encoding="utf-8").read()
s = sub_in_func(s, "void kernel_mul_mv_q5_K_f32_impl(", '[[host_name("kernel_mul_mv_q5_K_f32")]]', [
("""    const int first_row = (r0 * NSG + sgitg) * nr0;
""", """    // FC_mul_mv_split (host: rows of <= 2 super-blocks): the lanes with tiisg%4 >= 2, idle in the plain layout, take a
    // second group of nr0 rows; their partials go back to the plain layout's lane positions before the reduction
    const bool  dual = FC_mul_mv_split;
    const short grp  = dual && (tiisg%4) >= 2 ? 1 : 0;
    const int first_row = (r0 * NSG + sgitg) * nr0 * (dual ? 2 : 1) + grp*nr0;
"""),
("""    const short ix  = tiisg%4;
""", """    const short ix  = tiisg%4 - 2*grp;
"""),
("""    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = tot;
        }
    }
""", """    if (dual) {
        const int frA = first_row - grp*nr0;
        for (int row = 0; row < nr0; ++row) {
            const float totA = simd_sum(grp == 0 ? sumf[row] : 0.f);
            const float totB = simd_sum(simd_shuffle_xor(grp == 1 ? sumf[row] : 0.f, 2));
            if (tiisg == 0) {
                if (frA + row < args.ne0) {
                    dst_f32[frA + row] = totA;
                }
                if (frA + nr0 + row < args.ne0) {
                    dst_f32[frA + nr0 + row] = totB;
                }
            }
        }
        return;
    }

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = tot;
        }
    }
"""),
])
open(p, "w", encoding="utf-8", newline=chr(10)).write(s)

p = root + "/ggml/src/ggml-metal/ggml-metal-device.cpp"
s = open(p, encoding="utf-8").read()
old = """        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;"""
assert s.count(old) == 2, s.count(old)
i = s.index(old, s.index(old) + 1)   # the second one: get_pipeline_mul_mv_id
new = """        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
                // rows of <= 2 super-blocks (MoE down, K <= 512): the idle half of the lanes takes a second row group
                static const bool dual = !getenv("GGML_METAL_Q5K_DUAL") || atoi(getenv("GGML_METAL_Q5K_DUAL")) != 0;
                if (dual && ne00 <= 2*256) {
                    nr0 = 2*N_R0_Q5_K;
                    split = true;
                }
            } break;"""
s = s[:i] + new + s[i + len(old):]
open(p, "w", encoding="utf-8", newline=chr(10)).write(s)
print("ok")
