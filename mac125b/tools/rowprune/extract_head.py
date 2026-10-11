# Copy the target's final hyper-connection mixer (the model's output norm) out of the main GGUF into a small file for
# probe.py (logit-lens features). Python stdlib only (runs on the Mac without numpy). ~13 MB of output for bf16.
#
#   python3 extract_head.py ~/models/qwen38-gsq/IQ3_XXS/*.gguf -o ~/rowprune/head.bin
#
# Default tensors: output_hc_norm.weight, output_hc_down.weight, output_hc_up.weight (llama-arch names of
# LLM_TENSOR_HC_HEAD_NORM/DOWN/UP; qwen4exp builds result_norm = hc_mix(res_hc, these) and logits = output @ it).
# --names adds/replaces names. The output head itself (output.weight, Q5_K in the target) is NOT copied: probe.py uses
# the Q8_0 copy of the head in the MTP GGUF (same matrix, different quantization; probe.py checks the agreement at the
# last layer). Output format "RPHW": u32 n; per tensor: u32 len, name, u32 ggml_type, u32 nd, u64 dims[nd], u64 nbytes, raw.
import struct, sys, argparse

TYPE_SIZE = {0: (4, 1), 1: (2, 1), 30: (2, 1), 8: (34, 32)}   # ggml type -> (bytes per block, elements per block)

def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))

def rstr(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")

def skip(f, t):
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sz:
        f.read(sz[t])
    elif t == 8:
        rstr(f)
    elif t == 9:
        (at,) = rd(f, "I"); (n,) = rd(f, "Q")
        for _ in range(n):
            skip(f, at)
    else:
        raise ValueError("gguf kv type %d" % t)

def scan(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", path
    rd(f, "I"); (nt,) = rd(f, "Q"); (nkv,) = rd(f, "Q")
    align = 32
    for _ in range(nkv):
        k = rstr(f); (t,) = rd(f, "I")
        if k == "general.alignment":
            (align,) = rd(f, "I")
        else:
            skip(f, t)
    infos = []
    for _ in range(nt):
        name = rstr(f); (nd,) = rd(f, "I"); dims = rd(f, "Q" * nd); (tt,) = rd(f, "I"); (off,) = rd(f, "Q")
        infos.append((name, dims, tt, off))
    base = (f.tell() + align - 1) // align * align
    return f, base, infos

ap = argparse.ArgumentParser()
ap.add_argument("gguf", nargs="+")
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--names", default="output_hc_norm.weight,output_hc_down.weight,output_hc_up.weight")
a = ap.parse_args()
want = [n for n in a.names.split(",") if n]

found = {}
for p in a.gguf:
    f, base, infos = scan(p)
    for name, dims, tt, off in infos:
        if name in want and name not in found:
            if tt not in TYPE_SIZE:
                sys.exit("%s has ggml type %d; extend TYPE_SIZE here and the decoder in probe.py" % (name, tt))
            n = 1
            for d in dims:
                n *= d
            bs, be = TYPE_SIZE[tt]
            nbytes = n // be * bs
            f.seek(base + off)
            found[name] = (dims, tt, f.read(nbytes))
            print("%-28s %-22s type %2d %9d bytes  (%s)" % (name, dims, tt, nbytes, p))
missing = [n for n in want if n not in found]
if missing:
    sys.exit("missing tensors: %s" % missing)
with open(a.out, "wb") as o:
    o.write(b"RPHW"); o.write(struct.pack("<I", len(want)))
    for name in want:
        dims, tt, raw = found[name]
        nb = name.encode()
        o.write(struct.pack("<I", len(nb))); o.write(nb)
        o.write(struct.pack("<II", tt, len(dims))); o.write(struct.pack("<" + "Q" * len(dims), *dims))
        o.write(struct.pack("<Q", len(raw))); o.write(raw)
print("wrote", a.out)
