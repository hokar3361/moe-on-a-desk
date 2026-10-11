# write trained MTP parameters back into a copy of the MTP GGUF, in each tensor's original type (q8_0 / f32 / bf16)
# usage: export_mtp.py in.gguf trained.pt out.gguf
import sys, shutil, struct, numpy as np, torch
sys.path.insert(0, "$HOME/kbench")
from mtp_torch import _rd, _rstr, _skip

NAME = {  # module parameter -> gguf tensor
    "enorm": "nextn.enorm.weight", "hnorm": "nextn.hnorm.weight", "eh_proj": "nextn.eh_proj.weight",
    "hc_attn_norm": "hc_attn_norm.weight", "hc_attn_down": "hc_attn_down.weight", "hc_attn_up": "hc_attn_up.weight",
    "hc_attn_inject": "hc_attn_inject.weight", "hc_ffn_norm": "hc_ffn_norm.weight", "hc_ffn_down": "hc_ffn_down.weight",
    "hc_ffn_up": "hc_ffn_up.weight", "hc_ffn_inject": "hc_ffn_inject.weight",
    "hc_head_norm": "nextn.hc_head_norm.weight", "hc_head_down": "nextn.hc_head_down.weight", "hc_head_up": "nextn.hc_head_up.weight",
    "wq": "attn_q.weight", "wk": "attn_k.weight", "wv": "attn_v.weight", "wo": "attn_output.weight",
    "q_norm": "attn_q_norm.weight", "k_norm": "attn_k_norm.weight", "router": "ffn_gate_inp.weight",
    "up_s": "ffn_up_shexp.weight", "gate_s": "ffn_gate_shexp.weight", "down_s": "ffn_down_shexp.weight",
    "gate_inp_s": "ffn_gate_inp_shexp.weight",
}

src, ptf, dst = sys.argv[1], sys.argv[2], sys.argv[3]
shutil.copyfile(src, dst)
f = open(src, "rb")
assert f.read(4) == b"GGUF"
_rd(f, "I"); (nt,) = _rd(f, "Q"); (nkv,) = _rd(f, "Q")
align = 32
for _ in range(nkv):
    k = _rstr(f); (t,) = _rd(f, "I")
    if k == "general.alignment": (align,) = _rd(f, "I")
    else: _skip(f, t)
infos = {}
for _ in range(nt):
    name = _rstr(f); (nd,) = _rd(f, "I"); dims = _rd(f, "Q"*nd); (tt,) = _rd(f, "I"); (off,) = _rd(f, "Q")
    infos[name] = (dims, tt, off)
base = (f.tell() + align - 1)//align*align
sd = torch.load(ptf)
out = open(dst, "r+b")
for pn, t in sd.items():
    gn = "blk.48." + NAME[pn]
    dims, tt, off = infos[gn]
    a = t.float().numpy().reshape(-1)
    assert a.size == int(np.prod(dims)), (pn, a.size, dims)
    if tt == 0:
        b = a.astype(np.float32).tobytes()
    elif tt == 30:
        u = a.astype(np.float32).view(np.uint32)
        b = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
    elif tt == 8:
        x = a.reshape(-1, 32)
        amax = np.abs(x).max(1)
        dsc = (amax/127.0).astype(np.float32)
        inv = np.where(dsc > 0, 1.0/np.maximum(dsc, 1e-30), 0.0)
        q = np.clip(np.round(x*inv[:, None]), -127, 127).astype(np.int8)
        blk = np.zeros((x.shape[0], 34), dtype=np.uint8)
        blk[:, :2] = dsc.astype(np.float16).view(np.uint8).reshape(-1, 2)
        blk[:, 2:] = q.view(np.uint8)
        b = blk.tobytes()
    else:
        raise ValueError((gn, tt))
    out.seek(base + off)
    out.write(b)
    print("wrote", gn, tt, len(b))
out.close()
