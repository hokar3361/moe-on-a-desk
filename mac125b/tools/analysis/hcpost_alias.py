# HC_POST+norm fusion: the kernel reads the shared block output x from every stream's threadgroup, so its second output
# (norm*gamma) must not reuse the memory of x / residual / post. Without delayed frees the allocator may hand x's memory
# to that output (x is dead after HC_POST), and later threadgroups then read overwritten data (PPL 1.23 -> 253 at ub 512).
import sys
p = sys.argv[1] + "/ggml/src/ggml-metal/ggml-metal-ops.cpp"
s = open(p).read()
old = """static bool ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, const ggml_tensor * node) {"""
assert s.count(old) == 1
s = s.replace(old, """// true when the memory of a and b (their base tensors) overlaps
static bool ggml_metal_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const ggml_tensor * ba = a->view_src ? a->view_src : a;
    const ggml_tensor * bb = b->view_src ? b->view_src : b;
    if (ba->buffer != bb->buffer || !ba->data || !bb->data) {
        return ba == bb;
    }
    const uint8_t * a0 = (const uint8_t *) ba->data, * a1 = a0 + ggml_nbytes(ba);
    const uint8_t * b0 = (const uint8_t *) bb->data, * b1 = b0 + ggml_nbytes(bb);
    return a0 < b1 && b0 < a1;
}

""" + old)
old = """                ggml_is_contiguous(op) && ggml_is_contiguous(mul) && x->nb[0] == 4 && residual->nb[0] == 4 &&"""
assert s.count(old) == 1
s = s.replace(old, """                !ggml_metal_tensors_overlap(mul, x) && !ggml_metal_tensors_overlap(mul, residual) &&
                !ggml_metal_tensors_overlap(mul, inject) && !ggml_metal_tensors_overlap(mul, op) &&
                ggml_is_contiguous(op) && ggml_is_contiguous(mul) && x->nb[0] == 4 && residual->nb[0] == 4 &&""")
open(p, "w").write(s)
print("hcpost_alias ok")
