"""Noise-free DUST: backprop through attention with the credit DUST targets in expectation.

DUST's per-token rewards mean that, in expectation, the error reaching a residual-stream site at token t
is dL_t/dY_t: the own-token gradient. Across attention that keeps only the diagonal (query t reading
key/value t). Keys, values, value-embedding gates and value embeddings are credited separately,
against the o@ errors of every later query u, discounted by gamma^(u-t); that credit updates their
weights but does not flow further down the stream.

So each attention gets two gradient routes with the forward value unchanged:
  stream route  q: exact.  k, v -> residual stream: diagonal only ('own'), or exact ('full').
  weight route  k, v -> W_k, W_v, gate, value embedding: credit weighted by gamma^(u-t).
Everything else (MLPs, writers, head) is position-local and gets ordinary backprop.

  GPT stock DUST     every layer: stream own, gamma 0.98
  GPT + attn VJP     every layer: stream own, gamma 1
  YOCO stock DUST    all layers own, gamma 0.98 (the cache is credited like any k/v)
  YOCO + cache VJP   all layers own, gamma 1
  YOCO + handoff     self-decoder own, gamma 1; cross-decoder exact (stream full, gamma 1)
"""
import math
import types

import torch
from torch.nn import functional as F

from model import apply_rotary_emb, norm


def _split(lin, x):
    """Same value twice: the first carries gradient only to the weight, the second only to the input."""
    return F.linear(x.detach(), lin.weight), F.linear(x, lin.weight.detach())


def ed_attention(q, ks, vs, kw, vw, window, gamma, stream):
    """q, k, v: [B, T, H, D]. Returns [B, T, H, D] with the two-route gradient described above."""
    B, T, H, D = q.shape
    sc = 1 / math.sqrt(D)
    pos = torch.arange(T, device=q.device)
    lag = pos[:, None] - pos[None, :]                 # [u, s]
    mask = (lag >= 0) & (lag <= window)
    q, ks, vs, kw, vw = (t.float().transpose(1, 2) for t in (q, ks, vs, kw, vw))  # [B, H, T, D]

    def probs(qq, kk):
        return (qq @ kk.transpose(-1, -2) * sc).masked_fill(~mask, float('-inf')).softmax(-1)

    if stream == 'full':
        out = probs(q, ks) @ vs
    else:
        P = probs(q, ks.detach())
        out = P @ vs.detach()                         # exact gradient for q only
        Pd, od = P.detach(), out.detach()
        a = Pd.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)   # a_uu
        c = (q.detach() * ks).sum(-1, keepdim=True)        # q_u . k_u, gradient into k_u
        diag = a * vs + sc * a * c * (vs.detach() - od)
        out = out + diag - diag.detach()
    Pd, od = (probs(q, ks).detach(), (probs(q, ks) @ vs).detach()) if stream == 'full' else (Pd, od)
    G = torch.where(mask, gamma ** lag.clamp(min=0).float(), torch.zeros((), device=q.device))
    M = Pd * G                                        # [B, H, u, s]
    C = q.detach() @ kw.transpose(-1, -2)             # c_us = q_u . k_s, gradient into k_s
    MC = M * C
    Y = M @ vw + sc * (MC @ vs.detach() - MC.sum(-1, keepdim=True) * od)
    out = out + Y - Y.detach()
    return out.transpose(1, 2)


def _self_forward(self, x, ve, cos_sin, window_size):
    B, T, C = x.size()
    h, d = self.n_head, self.head_dim
    q = self.c_q(x).view(B, T, h, d)
    kw, ks = (t.view(B, T, h, d) for t in _split(self.c_k, x))
    vw, vs = (t.view(B, T, h, d) for t in _split(self.c_v, x))
    if ve is not None:
        ve = ve.view(B, T, h, d)
        gw, gs = _split(self.ve_gate, x[..., :self.ve_gate_channels])
        vw = vw + (2 * torch.sigmoid(gw)).unsqueeze(-1) * ve
        vs = vs + (2 * torch.sigmoid(gs)).unsqueeze(-1) * ve.detach()
    cos, sin = cos_sin
    pq = lambda t: norm(apply_rotary_emb(t, cos, sin))
    y = ed_attention(pq(q), pq(ks), vs, pq(kw), vw, window_size[0], self.ed_gamma, self.ed_stream)
    return self.c_proj(y.to(x.dtype).contiguous().view(B, T, -1))


def _cache(self, x, cos_sin):
    B, T, _ = x.shape
    h, d = self.config.n_kv_head, self.config.n_embd // self.config.n_head
    xn = norm(x)
    kw, ks = (t.view(B, T, h, d) for t in _split(self.transformer.cache_k, xn))
    vw, vs = (t.view(B, T, h, d) for t in _split(self.transformer.cache_v, xn))
    cos, sin = cos_sin
    pk = lambda t: norm(apply_rotary_emb(t, cos, sin))
    return (pk(ks), pk(kw)), (vs, vw)


def _cross_forward(self, x, k, v, cos_sin, window_size):
    B, T, C = x.size()
    cos, sin = cos_sin
    q = norm(apply_rotary_emb(self.c_q(x).view(B, T, self.n_head, self.head_dim), cos, sin))
    y = ed_attention(q, k[0], v[0], k[1], v[1], window_size[0], self.ed_gamma, self.ed_stream)
    return self.c_proj(y.to(x.dtype).contiguous().view(B, T, -1))


def install(model, gamma, cross_full=False):
    """Swap every attention forward for its expected-DUST version."""
    yoco = hasattr(model, 'n_self')
    n_self = model.n_self if yoco else model.config.n_layer
    for i, block in enumerate(model.transformer.h):
        attn = block.attn
        if i < n_self:
            attn.ed_gamma, attn.ed_stream = gamma, 'own'
            attn.forward = types.MethodType(_self_forward, attn)
        else:
            attn.ed_gamma, attn.ed_stream = (1.0, 'full') if cross_full else (gamma, 'own')
            attn.forward = types.MethodType(_cross_forward, attn)
    if yoco:
        model.cache = types.MethodType(_cache, model)
    return model
