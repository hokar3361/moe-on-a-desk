# PyTorch re-implementation of the qwen4exp MTP block (llama.cpp src/models/qwen4exp.cpp graph_mtp), dense-attention
# regime (sequence <= indexer top_k, where QSA selects every block). Weights are read from the MTP GGUF.
import struct, numpy as np, torch, torch.nn.functional as F

NE, HC, LR = 2560, 4, 320
NH, NKV, HD, NROT = 24, 2, 256, 64
NEXP, NUSED, NFF = 512, 10, 640
EPS = 1e-6
ROPE_BASE = 1e7

def _rd(f, fmt):
    n = struct.calcsize(fmt); return struct.unpack("<" + fmt, f.read(n))
def _rstr(f):
    (n,) = _rd(f, "Q"); return f.read(n).decode("utf-8", "replace")
def _skip(f, t):
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in sz: f.read(sz[t])
    elif t == 8: _rstr(f)
    elif t == 9:
        (at,) = _rd(f, "I"); (n,) = _rd(f, "Q")
        for _ in range(n): _skip(f, at)

def load_gguf(path, want=None):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    _rd(f, "I"); (nt,) = _rd(f, "Q"); (nkv,) = _rd(f, "Q")
    align = 32
    for _ in range(nkv):
        k = _rstr(f); (t,) = _rd(f, "I")
        if k == "general.alignment": (align,) = _rd(f, "I")
        else: _skip(f, t)
    infos = []
    for _ in range(nt):
        name = _rstr(f); (nd,) = _rd(f, "I"); dims = _rd(f, "Q"*nd); (tt,) = _rd(f, "I"); (off,) = _rd(f, "Q")
        infos.append((name, dims, tt, off))
    base = (f.tell() + align - 1)//align*align
    out = {}
    for name, dims, tt, off in infos:
        if want is not None and not want(name):
            continue
        n = int(np.prod(dims))
        f.seek(base + off)
        if tt == 0:      # f32
            a = np.frombuffer(f.read(4*n), dtype=np.float32)
        elif tt == 30:   # bf16
            a = (np.frombuffer(f.read(2*n), dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif tt == 1:    # f16
            a = np.frombuffer(f.read(2*n), dtype=np.float16).astype(np.float32)
        elif tt == 8:    # q8_0: [d fp16][32 x int8]
            nb = n//32
            raw = np.frombuffer(f.read(34*nb), dtype=np.uint8).reshape(nb, 34)
            d = raw[:, :2].copy().view(np.float16).astype(np.float32)
            q = raw[:, 2:].view(np.int8).astype(np.float32)
            a = (q*d).reshape(-1)
        else:
            raise ValueError("type %d for %s" % (tt, name))
        out[name] = torch.from_numpy(a.reshape(tuple(reversed(dims))).copy())
    return out

def rms(x, w=None):
    y = x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y*w if w is not None else y

def rope_neox(x, pos):
    # x: [T, H, HD]; rotate the first NROT dims, pairs (i, i + NROT/2)
    half = NROT//2
    inv = ROPE_BASE ** (-torch.arange(0, half, dtype=torch.float32, device=x.device)*2.0/NROT)
    ang = pos.float()[:, None]*inv[None, :]
    c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    x1, x2, rest = x[..., :half], x[..., half:NROT], x[..., NROT:]
    return torch.cat([x1*c - x2*s, x2*c + x1*s, rest], dim=-1)

class MTP(torch.nn.Module):
    def __init__(self, W, device="mps", dtype=torch.float32, expert_dtype=torch.bfloat16):
        super().__init__()
        g = lambda k: W[k]
        p = lambda t, dt=dtype: torch.nn.Parameter(t.to(device=device, dtype=dt))
        L = "blk.48."
        self.tok_embd = g("token_embd.weight").to(device=device, dtype=expert_dtype)   # [V, NE], frozen
        self.out_w    = g("output.weight").to(device=device, dtype=expert_dtype)       # [V, NE], frozen
        self.enorm = p(g(L + "nextn.enorm.weight")); self.hnorm = p(g(L + "nextn.hnorm.weight").reshape(HC, NE))
        self.eh_proj = p(g(L + "nextn.eh_proj.weight"))                                 # [NE, 2*NE]
        for k in ["attn", "ffn"]:
            setattr(self, "hc_%s_norm" % k, p(g(L + "hc_%s_norm.weight" % k).reshape(HC, NE)))
            setattr(self, "hc_%s_down" % k, p(g(L + "hc_%s_down.weight" % k)))             # [LR, HC*NE]
            setattr(self, "hc_%s_up" % k, p(g(L + "hc_%s_up.weight" % k)))                 # [HC*NE, LR]
            setattr(self, "hc_%s_inject" % k, p(g(L + "hc_%s_inject.weight" % k)))         # [HC, HC*NE]
        self.hc_head_norm = p(g(L + "nextn.hc_head_norm.weight").reshape(HC, NE))
        self.hc_head_down = p(g(L + "nextn.hc_head_down.weight"))
        self.hc_head_up   = p(g(L + "nextn.hc_head_up.weight"))
        self.wq = p(g(L + "attn_q.weight")); self.wk = p(g(L + "attn_k.weight")); self.wv = p(g(L + "attn_v.weight"))
        self.wo = p(g(L + "attn_output.weight"))
        self.q_norm = p(g(L + "attn_q_norm.weight")); self.k_norm = p(g(L + "attn_k_norm.weight"))
        self.router = p(g(L + "ffn_gate_inp.weight"))                                   # [NEXP, NE]
        self.up_e   = p(g(L + "ffn_up_exps.weight"), expert_dtype)                      # [NEXP, NFF, NE]
        self.gate_e = p(g(L + "ffn_gate_exps.weight"), expert_dtype)
        self.down_e = p(g(L + "ffn_down_exps.weight"), expert_dtype)                    # [NEXP, NE, NFF]
        self.up_s = p(g(L + "ffn_up_shexp.weight")); self.gate_s = p(g(L + "ffn_gate_shexp.weight"))
        self.down_s = p(g(L + "ffn_down_shexp.weight")); self.gate_inp_s = p(g(L + "ffn_gate_inp_shexp.weight"))

    def hc_mix(self, res, wn, wd, wu, wi):
        # res [T, HC, NE] -> mixed [T, NE], inject [T, HC] (or None)
        T = res.shape[0]
        xn = rms(res)*wn
        flat = xn.reshape(T, HC*NE)
        inject = flat @ wi.t() if wi is not None else None
        lo = F.silu((flat @ wd.t())/HC)
        gate = (lo @ wu.t()).reshape(T, HC, NE)
        mixed = (xn*torch.sigmoid(gate)).mean(1)
        return mixed, inject

    @staticmethod
    def hc_combine(res, out, inject):
        w = 2.0*torch.sigmoid(inject/HC)                     # [T, HC]
        return res + w[:, :, None]*out[:, None, :]

    def attn(self, x, pos):
        T = x.shape[0]
        qf = (x @ self.wq.t()).reshape(T, NH, 2*HD)
        q, gate = qf[..., :HD], qf[..., HD:]
        q = rms(q, self.q_norm)
        k = rms((x @ self.wk.t()).reshape(T, NKV, HD), self.k_norm)
        v = (x @ self.wv.t()).reshape(T, NKV, HD)
        q, k = rope_neox(q, pos), rope_neox(k, pos)
        k = k.repeat_interleave(NH//NKV, dim=1); v = v.repeat_interleave(NH//NKV, dim=1)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True, scale=1.0/HD**0.5)
        o = o.transpose(0, 1).reshape(T, NH*HD)*torch.sigmoid(gate.reshape(T, NH*HD))
        return o @ self.wo.t()

    def moe(self, x):
        T = x.shape[0]
        probs = torch.softmax(x @ self.router.t(), dim=-1)
        w, idx = torch.topk(probs, NUSED, dim=-1)
        w = w/w.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        flat_idx = idx.reshape(-1)
        tok = torch.arange(T, device=x.device).repeat_interleave(NUSED)
        wflat = w.reshape(-1)
        for e in torch.unique(flat_idx).tolist():
            m = flat_idx == e
            xt = x[tok[m]].to(self.up_e.dtype)
            hdn = F.silu(xt @ self.gate_e[e].t())*(xt @ self.up_e[e].t())
            out.index_add_(0, tok[m], (hdn @ self.down_e[e].t()).float()*wflat[m][:, None])
        sh = (F.silu(x @ self.gate_s.t())*(x @ self.up_s.t())) @ self.down_s.t()
        sh = sh*torch.sigmoid(x @ self.gate_inp_s.reshape(1, -1).t())
        return out + sh

    def forward(self, tokens, h_in, pos):
        # tokens [T] (token at position pos), h_in [T, HC*NE] (target/previous hidden), pos [T]
        T = tokens.shape[0]
        e = self.tok_embd[tokens].float()
        e_n = rms(e, self.enorm)[:, None, :].expand(T, HC, NE)
        h_n = rms(h_in.reshape(T, HC, NE), self.hnorm)
        res = torch.cat([e_n, h_n], dim=-1) @ self.eh_proj.t()        # [T, HC, NE]
        mixed, inj = self.hc_mix(res, self.hc_attn_norm, self.hc_attn_down, self.hc_attn_up, self.hc_attn_inject)
        res = self.hc_combine(res, self.attn(mixed, pos), inj)
        mixed, inj = self.hc_mix(res, self.hc_ffn_norm, self.hc_ffn_down, self.hc_ffn_up, self.hc_ffn_inject)
        res = self.hc_combine(res, self.moe(mixed), inj)
        h_out = res.reshape(T, HC*NE)
        head, _ = self.hc_mix(res, self.hc_head_norm, self.hc_head_down, self.hc_head_up, None)
        return h_out, head                                              # logits = head @ out_w.T

    def logits(self, head):
        return head.to(self.out_w.dtype) @ self.out_w.t()
