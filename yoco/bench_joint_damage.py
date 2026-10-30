"""Joint noise damage of writer updates: own-only vs pooled estimates.

Two independent full sets of writer (+ embedding) estimates give one noise sample per site, eps = (E1 - E2)/sqrt(2),
for each method. Applying all sites' noise at once as an update (lr / (1 - beta) / tokens) and taking the loss second
difference on validation measures the damage in function space, where noise correlated across depths adds up.
Also printed: the summed per-site weight-space noise energy, the quantity per-site error metrics see.
"""
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import make_optimizer, recipe  # noqa: E402
from yoco.bench_local_zo import load  # noqa: E402
from yoco.dustx import DustX  # noqa: E402

run, R = sys.argv[1], int(sys.argv[2])
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')
model = load(run)
L = model.config.n_layer
st = recipe('10M', 256)
opt = make_optimizer(model, st)
lr_of = {id(p): g['lr'] for g in opt.param_groups for p in g['params']}
params = dict(model.named_parameters())
data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
val = torch.load('data/tinystories/validation.pt', weights_only=True).long()[:64].cuda()
tokens = (y != -1).sum().item()


@torch.no_grad()
def vloss():
    return sum(model(b[:, :-1].contiguous(), b[:, 1:].contiguous(), loss_reduction='sum').item()
               for b in val.split(32)) / (val.shape[0] * (val.shape[1] - 1))


est = DustX(model, st, seed=9)
est.pool = True
est.set_precision('fast16')
sites = [(f'transformer.h.{i}.attn.c_proj', f'transformer.h.{i}.mlp.c_proj', st.writers[i]) for i in range(L)]
with torch.no_grad():
    est.capture(x, y)
    L0 = vloss()

    def one():
        e = {}
        for a, b, n in sites:
            e.update(est.direct_errors([a, b], n, 0.2))
        e.update(est.direct_errors(['transformer.wte'], st.embedding, 0.2))
        pooled = est.pool_writers(e)
        est._halves = {}
        return ({s: est.reassemble(s, v) for s, v in e.items()}, {s: est.reassemble(s, v) for s, v in pooled.items()})

    for _ in range(4):  # warm the pooling moments
        one()
    res = {'own': [0.0, 0.0], 'pooled': [0.0, 0.0]}
    for r in range(R):
        a1, p1 = one()
        a2, p2 = one()
        for name, (G1, G2) in (('own', (a1, a2)), ('pooled', (p1, p2))):
            eps = {s: (G1[s] - G2[s]) / math.sqrt(2) for s in G1}
            energy = sum(v.square().sum().item() for v in eps.values())
            orig = {s: params[s + '.weight'].detach().clone() for s in eps}
            vals = []
            for sgn in (1, -1):
                for s in eps:
                    p = params[s + '.weight']
                    p.copy_(orig[s] + sgn * lr_of[id(p)] / (1 - st.momentum) / tokens * eps[s].to(p.dtype))
                vals.append(vloss())
            for s in eps:
                params[s + '.weight'].copy_(orig[s])
            res[name][0] += ((vals[0] + vals[1]) / 2 - L0) / R
            res[name][1] += energy / R
    for name, (dmg, en) in res.items():
        print(f'{name:7s} joint damage {dmg:+.5f}   summed per-site noise energy {en:.3e}', flush=True)
