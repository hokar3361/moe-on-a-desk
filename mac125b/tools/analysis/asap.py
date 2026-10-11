# GGML_METAL_REORDER_ASAP=1: list-schedule the (fused) nodes by their earliest stage (ASAP level), ties in graph order.
# level(i) = max over earlier conflicting nodes j (RAW/WAR/WAW by the same memory-range rules as the encoder) of
# level(j) + (j computes ? 1 : 0). Nodes whose op is not "safe" keep their relative order with everything (they act as
# fences: their level is at least the max level so far, and later nodes start at least at their level).
import sys
p = sys.argv[1] + "/ggml/src/ggml-metal/ggml-metal-common.cpp"
s = open(p).read()
if "GGML_METAL_REORDER_ASAP" not in s:
    old = """#if 1
    // reorder to improve concurrency
    const auto order = ggml_metal_graph_optimize_reorder(nodes);
#else"""
    assert s.count(old) == 1
    new = """#if 1
    // reorder to improve concurrency
    static const bool asap = getenv("GGML_METAL_REORDER_ASAP") != nullptr && atoi(getenv("GGML_METAL_REORDER_ASAP")) != 0;
    const auto order = asap ? ggml_metal_graph_optimize_asap(nodes) : ggml_metal_graph_optimize_reorder(nodes);
#else"""
    s = s.replace(old, new)
    anchor = "static std::vector<int> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {"
    assert s.count(anchor) == 1
    fn = """static std::vector<int> ggml_metal_graph_optimize_asap(const std::vector<node_info> & nodes) {
    const int n = nodes.size();

    // per node: its memory ranges (src and dst) as a tiny mem-ranges set, to test pairwise conflicts
    std::vector<ggml_mem_ranges_t> mr(n);
    for (int i = 0; i < n; ++i) {
        mr[i] = ggml_mem_ranges_init(0);
        const auto & nd = nodes[i];
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            const ggml_tensor * src = nd.node->src[k];
            if (src && !nd.is_output(src)) ggml_mem_ranges_add_src(mr[i], src);
        }
        for (const auto * f : nd.fused) {
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                const ggml_tensor * src = f->src[k];
                if (src && !nd.is_output(src)) ggml_mem_ranges_add_src(mr[i], src);
            }
        }
        ggml_mem_ranges_add_dst(mr[i], nd.node);
        for (const auto * f : nd.fused) ggml_mem_ranges_add_dst(mr[i], f);
    }

    // conflict(j, i): i reads what j writes, writes what j reads or writes
    const auto conflict = [&](int j, int i) {
        const auto & ni = nodes[i];
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            const ggml_tensor * src = ni.node->src[k];
            if (src && !ni.is_output(src) && !ggml_mem_ranges_check_src(mr[j], src)) return true;
        }
        for (const auto * f : ni.fused) {
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                const ggml_tensor * src = f->src[k];
                if (src && !ni.is_output(src) && !ggml_mem_ranges_check_src(mr[j], src)) return true;
            }
        }
        if (!ggml_mem_ranges_check_dst(mr[j], ni.node)) return true;
        for (const auto * f : ni.fused) if (!ggml_mem_ranges_check_dst(mr[j], f)) return true;
        return false;
    };

    const auto safe = [](ggml_op op) {
        switch (op) {
            case GGML_OP_MUL_MAT: case GGML_OP_MUL_MAT_ID: case GGML_OP_ROPE: case GGML_OP_NORM: case GGML_OP_RMS_NORM:
            case GGML_OP_GROUP_NORM: case GGML_OP_L2_NORM: case GGML_OP_SUM_ROWS: case GGML_OP_SSM_CONV: case GGML_OP_SSM_SCAN:
            case GGML_OP_CLAMP: case GGML_OP_TRI: case GGML_OP_DIAG: case GGML_OP_MUL: case GGML_OP_ADD: case GGML_OP_SUB:
            case GGML_OP_DIV: case GGML_OP_GLU: case GGML_OP_SCALE: case GGML_OP_UNARY: case GGML_OP_GET_ROWS: case GGML_OP_SET_ROWS:
            case GGML_OP_SET: case GGML_OP_CPY: case GGML_OP_CONT: case GGML_OP_REPEAT: case GGML_OP_CONCAT: case GGML_OP_SOFT_MAX:
            case GGML_OP_DSV4_HC_PRE: case GGML_OP_DSV4_HC_POST: case GGML_OP_FILL: case GGML_OP_CUMSUM: case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: case GGML_OP_GATED_DELTA_NET: case GGML_OP_FLASH_ATTN_EXT: case GGML_OP_LIGHTNING_INDEXER:
                return true;
            default:
                return ggml_op_is_empty(op);
        }
    };

    static const int window = getenv("GGML_METAL_REORDER_ASAP_WIN") ? atoi(getenv("GGML_METAL_REORDER_ASAP_WIN")) : (1 << 30);

    std::vector<int> lvl(n, 0);
    int fence = 0;      // level floor set by the last unsafe node
    int maxl  = 0;
    for (int i = 0; i < n; ++i) {
        int l = fence;
        if (!safe(nodes[i].op())) {
            l = std::max(l, maxl + 1);
        }
        for (int j = std::max(0, i - window); j < i; ++j) {
            if (conflict(j, i)) {
                l = std::max(l, lvl[j] + (nodes[j].is_empty() ? 0 : 1));
            }
        }
        lvl[i] = l;
        maxl = std::max(maxl, l);
        if (!safe(nodes[i].op())) {
            fence = l + 1;
        }
    }

    std::vector<int> res(n);
    for (int i = 0; i < n; ++i) res[i] = i;
    std::stable_sort(res.begin(), res.end(), [&](int a, int b) { return lvl[a] < lvl[b]; });

    for (int i = 0; i < n; ++i) ggml_mem_ranges_free(mr[i]);

    return res;
}

"""
    s = s.replace(anchor, fn + anchor)
open(p, "w").write(s)
print("asap ok")
