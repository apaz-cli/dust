"""How concentrated are the per-token errors at each Dust site? (Plan item 5, aligned subspaces.)

Dust perturbs a site's output, d numbers per token. If most of the error (the loss gradient at that output) lies in
a few directions, perturbing only along those directions cuts the noise by about f * d / r, where f is the share of
the error's energy in the r-dimensional subspace. The subspace has to be found in advance, so it is fit on some
batches and its share f measured on others.
The errors are true gradients at each site's output (autograd, a measuring stick only), on a backprop-trained model.
For each site: the top-r principal directions of the errors over the first half of the batches (uncentered), and
the share of the held-out errors' energy they capture, at r = 1, 4, 16, 64, 256; and the gain f * d / r.
Logged keys: <site>/r<r>/share, <site>/r<r>/gain, <site>/d.
usage: bench_spectrum.py RUN DATA_DIR [BATCHES]
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoco.bench_local_zo import load  # noqa: E402
from mlsweep.logger import MLSweepLogger  # noqa: E402  (no-op outside an mlsweep worker)

run, data_dir = sys.argv[1], sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 16
torch.manual_seed(0)
model = load(run)
L = model.config.n_layer
data = torch.load(f'{data_dir}/train.pt', weights_only=True).long()[:8 * B].cuda()
mods = {'wte': model.transformer.wte}
for i in range(L):
    blk = model.transformer.h[i]
    mods[f'h.{i}.attn.c_proj'] = blk.attn.c_proj
    mods[f'h.{i}.mlp.c_proj'] = blk.mlp.c_proj
    mods[f'h.{i}.mlp.c_fc'] = blk.mlp.c_fc

grabbed = {}


def hook(name):
    def f(mod, inp, out):
        out.retain_grad()
        grabbed[name] = out
    return f


hooks = [m.register_forward_hook(hook(n)) for n, m in mods.items()]
model.requires_grad_(True)
fit = {n: None for n in mods}   # uncentered second moments, first half of the batches
held = {n: [] for n in mods}    # held-out errors
for bi in range(B):
    rows = data[8 * bi:8 * (bi + 1)]
    x, y = (rows[:, :-1] % 8192).contiguous(), rows[:, 1:].contiguous()
    y = torch.where(y >= 8192, torch.full_like(y, -1), y)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        model(x, y, loss_reduction='sum').backward()
    for n in mods:
        g = grabbed[n].grad.float().reshape(-1, grabbed[n].shape[-1])
        if bi < B // 2:
            C = g.T @ g
            fit[n] = C if fit[n] is None else fit[n] + C
        else:
            held[n].append(g)
    model.zero_grad(set_to_none=True)
for h in hooks:
    h.remove()

record = {}
for n in mods:
    _, vecs = torch.linalg.eigh(fit[n].double())
    vecs = vecs.flip(-1).float()               # descending
    G = torch.cat(held[n])
    total = G.square().sum().item()
    d = G.shape[-1]
    record[f'{n}/d'] = d
    line = []
    for r in (1, 4, 16, 64, 256):
        if r > d:
            continue
        share = (G @ vecs[:, :r]).square().sum().item() / total
        record[f'{n}/r{r}/share'] = share
        record[f'{n}/r{r}/gain'] = share * d / r
        line.append(f'r{r} {share:.3f} (x{share * d / r:.1f})')
    print(f'{n:16s} d={d:5d}  ' + '  '.join(line), flush=True)
with MLSweepLogger() as logger:
    logger.log(record, step=1)
