# sanity checks for s0model: (1) stage 1 == MTP.forward (causal), cosine vs the llama.cpp dump;
# (2) brute force: stage j at s+j-1 recomputed one position at a time with explicit key lists equals the batched unroll.
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s0common import *
from s0model import unroll, stage
from s0eval import load_model
from mtp_torch import rms, rope_neox, NH, NKV, HD

m = load_model(os.path.join(MTPDIR, "mtp-Q8_0.gguf"), "-")
d, tok, segs = segments(os.path.join(MTPDIR, "pairs2_held.bin"))
a, e, ans, hdr = segs[6]
t = torch.from_numpy(tok[a:e].astype(np.int64)).cuda()
h = torch.from_numpy(np.asarray(d["ht"][a:e]).astype(np.float32)).cuda()
hm = torch.from_numpy(np.asarray(d["hm"][a:e]).astype(np.float32)).cuda()

def one(tokid, hin, p, Ks, Vs):
    """single position p with explicit past keys Ks/Vs ([n, NKV, HD]); returns h_out, head, k, v"""
    e_ = m.tok_embd[tokid.view(1)].float()
    e_n = rms(e_, m.enorm)[:, None, :].expand(1, HC, NE)
    h_n = rms(hin.reshape(1, HC, NE), m.hnorm)
    res = torch.cat([e_n, h_n], -1) @ m.eh_proj.t()
    x, inj = m.hc_mix(res, m.hc_attn_norm, m.hc_attn_down, m.hc_attn_up, m.hc_attn_inject)
    qf = (x @ m.wq.t()).reshape(1, NH, 2*HD); q, gate = qf[..., :HD], qf[..., HD:]
    q = rms(q, m.q_norm); k = rms((x @ m.wk.t()).reshape(1, NKV, HD), m.k_norm); v = (x @ m.wv.t()).reshape(1, NKV, HD)
    pp = torch.tensor([p], device="cuda"); q, k = rope_neox(q, pp), rope_neox(k, pp)
    K = torch.cat([Ks, k], 0).repeat_interleave(NH//NKV, 1); V = torch.cat([Vs, v], 0).repeat_interleave(NH//NKV, 1)
    sc = torch.einsum("hd,nhd->hn", q[0], K)/HD**0.5
    o = torch.einsum("hn,nhd->hd", torch.softmax(sc, -1), V).reshape(1, NH*HD)*torch.sigmoid(gate.reshape(1, NH*HD))
    res = m.hc_combine(res, o @ m.wo.t(), inj)
    x, inj = m.hc_mix(res, m.hc_ffn_norm, m.hc_ffn_down, m.hc_ffn_up, m.hc_ffn_inject)
    res = m.hc_combine(res, m.moe(x), inj)
    head, _ = m.hc_mix(res, m.hc_head_norm, m.hc_head_down, m.hc_head_up, None)
    return res.reshape(1, NEF), head, k, v

with torch.no_grad():
    L = 300
    t, h, hm = t[:L], h[:L], hm[:L]
    pos = torch.arange(L, device="cuda")
    hin = torch.cat([torch.zeros(1, NEF, device="cuda"), h[:-1]], 0)
    ho_ref, hd_ref = m(t, hin, pos)
    ho, hd, K1, V1 = stage(m, t, hin, pos, 1, [])
    print("stage1 vs MTP.forward: max abs diff h %.3g head %.3g" % ((ho - ho_ref).abs().max().item(), (hd - hd_ref).abs().max().item()))
    a_, b_ = ho[1:], hm[1:]
    print("stage1 cosine vs llama.cpp h_mtp: %.6f" % ((a_*b_).sum(1)/(a_.norm(dim=1)*b_.norm(dim=1))).mean().item())
    heads = {j: hh for j, hh in unroll(m, t, h, 4, "inf")}
    for s in (40, 150, 290):
        # explicit chain: stage 1 at s uses K1[0..s-1] (+ own), stage i>1 at s+i-1 uses K1[0..s], chain keys, own
        Ks, Vs = K1[:s], V1[:s]
        hprev = h[s - 1:s]
        for j in range(1, 5):
            p = s + j - 1
            if p >= L: break
            hout, head, k, v = one(t[p], hprev, p, Ks, Vs)
            ref = heads[j][p:p + 1]
            print("s=%d stage %d: max abs diff head %.3g (norm %.1f)" % (s, j, (head - ref).abs().max().item(), ref.norm().item()))
            Ks, Vs = torch.cat([Ks, k], 0), torch.cat([Vs, v], 0)
            hprev = hout
