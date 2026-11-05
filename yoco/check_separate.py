"""Check DustX.separate_scores against brute force: perturb one key (or value) at a time, rerun softmax attention."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoco.dustx import DustX  # noqa: E402

torch.manual_seed(0)
n, b, t, h, d, window, g = 2, 1, 12, 2, 8, 6, 0.9
qp, kp, v, eo = (torch.randn(b, t, h, d, dtype=torch.float64) for _ in range(4))
pos = torch.arange(t)
lag_us = pos[:, None] - pos[None, :]
mask = (lag_us >= 0) & (lag_us <= window)
credit = torch.where(mask.T, g ** (-lag_us.T).clamp(min=0).double(), torch.zeros((), dtype=torch.float64))  # [s,u]


def attn(q, k, vv):
    L = torch.einsum('buhd,bshd->bhus', q, k) / d ** 0.5
    P = L.masked_fill(~mask, float('-inf')).softmax(-1)
    return P, torch.einsum('bhus,bshd->buhd', P, vv)


P, o = attn(qp, kp, v)
dk = 0.3 * torch.randn(n, b, t, h, d, dtype=torch.float64)
dv = 0.3 * torch.randn(n, b, t, h, d, dtype=torch.float64)
fast_k = DustX.separate_scores('k', P, eo, v, o, credit, dk=dk, qp=qp)
fast_v = DustX.separate_scores('v', P, eo, v, o, credit, dv=dv)
slow_k, slow_v = torch.zeros_like(fast_k), torch.zeros_like(fast_v)
for j in range(n):
    for s in range(t):
        k2 = kp.clone(); k2[:, s] += dk[j, :, s]
        v2 = v.clone(); v2[:, s] += dv[j, :, s]
        for kind, (kk, vv), out in (('k', (k2, v), slow_k), ('v', (kp, v2), slow_v)):
            _, o2 = attn(qp, kk, vv)
            sc = torch.einsum('buhd,buhd->bhu', eo, o2 - o)          # [b,h,u]
            out[j, :, s] = torch.einsum('bhu,u->bh', sc, credit[s])
for name, f, sl in (('k', fast_k, slow_k), ('v', fast_v, slow_v)):
    print(name, 'max abs err', (f - sl).abs().max().item(), 'scale', sl.abs().max().item())
