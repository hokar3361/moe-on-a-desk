# Draft-only coarse LM head for the qwen35moe MTP block (as qwen4exp): a low-bit copy of output (LLAMA_MTP_COARSE_HEAD, default
# q1_0, "off" disables) gives the draft's full-vocabulary scores; its top 32 are re-scored with the real head rows and every
# other logit is -inf. Only the MTP draft graph (one row) uses it; the target's verify pass is unchanged.
import sys
root = sys.argv[1]
def edit(path, pairs):
    p = root + "/" + path
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:60])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline=chr(10)).write(s)

edit("src/llama-model.cpp", [(
"""    if (arch == LLM_ARCH_QWEN4EXP && output && output->type == GGML_TYPE_Q8_0) {""",
"""    if ((arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_QWEN35MOE) && output && output->type == GGML_TYPE_Q8_0) {""")])

edit("src/models/qwen35moe.cpp", [(
"""    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35MOE MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);""",
"""    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35MOE MTP: missing LM head (nextn.shared_head_head or model.output)");
    if (model.output_coarse && head_w == model.output && !head_s && cur->ne[1] == 1) {
        // draft readout: coarse full-vocab head -> top-32 -> exact re-score with the real head rows; every other logit is -inf
        const int64_t n_vocab = model.output->ne[1];
        ggml_tensor * coarse = ggml_mul_mat(ctx0, model.output_coarse, cur);          // [n_vocab, 1]
        ggml_tensor * idx    = ggml_top_k(ctx0, coarse, 32);                          // [32, 1]
        ggml_tensor * rows   = ggml_get_rows(ctx0, model.output, idx);                // [n_embd, 32]
        ggml_tensor * exact  = ggml_mul_mat(ctx0, rows, cur);                         // [32, 1]
        ggml_tensor * full   = ggml_fill(ctx0, coarse, -INFINITY);                    // [n_vocab, 1]
        full = ggml_set_rows(ctx0, ggml_reshape_2d(ctx0, full, 1, n_vocab), ggml_reshape_2d(ctx0, exact, 1, 32), idx);
        cur  = ggml_reshape_2d(ctx0, full, n_vocab, 1);
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }""")])
print("ok")
