# graph reorder: more ops may be pulled forward into the current concurrent set (their memory access is fully described
# by src/dst ranges), and a longer look-ahead (GGML_METAL_REORDER_FWD, default 64 upstream)
import sys
p = sys.argv[1] + "/ggml/src/ggml-metal/ggml-metal-common.cpp"
s = open(p).read()
old = """            case GGML_OP_REPEAT:
                return true;
            default:
                return ggml_op_is_empty(op);"""
assert s.count(old) == 1
new = """            case GGML_OP_REPEAT:
                return true;
            // GGML_METAL_REORDER_EXT=1: ops of the hybrid models whose memory access is fully described by src/dst
            case GGML_OP_CONCAT:
            case GGML_OP_SOFT_MAX:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST:
            case GGML_OP_FILL:
            case GGML_OP_CUMSUM:
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K:
            case GGML_OP_GATED_DELTA_NET:
            case GGML_OP_FLASH_ATTN_EXT:
            case GGML_OP_LIGHTNING_INDEXER:
                {
                    static const bool ext = getenv("GGML_METAL_REORDER_EXT") != nullptr && atoi(getenv("GGML_METAL_REORDER_EXT")) != 0;
                    return ext;
                }
            default:
                return ggml_op_is_empty(op);"""
s = s.replace(old, new)
old = "            constexpr int N_FORWARD = 64;"
assert s.count(old) == 1
s = s.replace(old, "            static const int N_FORWARD = getenv(\"GGML_METAL_REORDER_FWD\") ? atoi(getenv(\"GGML_METAL_REORDER_FWD\")) : 64;")
open(p, "w").write(s)
print("reorder ok")
