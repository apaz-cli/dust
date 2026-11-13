"""Residual-stream draws with attention held clean (--freeze-attn) vs stock draws, at equal draws.

Holding every later attention output (and YOCO's cache) at its clean value makes a perturbation of one token's
residual stream reach only that token's own loss: no leakage, but every path through later attention is lost
(a bias). Scored against the true weight gradient from autograd (a measuring stick only): cosine and relative
error ||G_est - G_true||^2 / ||G_true||^2, per site, at N draws, averaged over R repeats. A large N shows each
method's bias floor. Logged keys: <mode>/<site>/N<draws>/<cos|relerr>.
usage: bench_freeze.py RUN DATA_DIR [R]
"""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import recipe  # noqa: E402
from yoco.bench_local_zo import load  # noqa: E402
from yoco.dustx import DustX  # noqa: E402
from mlsweep.logger import MLSweepLogger  # noqa: E402  (no-op outside an mlsweep worker)

run, data_dir = sys.argv[1], sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 3
torch.manual_seed(0)
model = load(run)
L = model.config.n_layer
data = torch.load(f'{data_dir}/train.pt', weights_only=True).long()[:8].cuda()
x, y = (data[:, :-1] % 8192).contiguous(), data[:, 1:].contiguous()
y = torch.where(y >= 8192, torch.full_like(y, -1), y)

# true gradients of the site weights (backprop, measuring stick only)
groups = [[f'transformer.h.{i}.attn.c_proj', f'transformer.h.{i}.mlp.c_proj'] for i in range(L)] + \
         [[f'transformer.h.{i}.mlp.c_fc'] for i in range(L)] + [['transformer.wte']]
sites = [s for g in groups for s in g]
params = dict(model.named_parameters())
model.requires_grad_(True)
with torch.autocast('cuda', dtype=torch.bfloat16):
    model(x, y, loss_reduction='sum').backward()
G_true = {s: params[s + '.weight'].grad.float().clone() for s in sites}
model.zero_grad(set_to_none=True)
model.requires_grad_(False)

est = DustX(model, recipe('10M', 256), seed=11)
est.set_precision('fast16')
record = {}
with torch.no_grad():
    est.capture(x, y)
    for N in (16, 64, 256, 1024):
        reps = R if N < 1024 else 1
        for mode in ('stock', 'freeze'):
            est.freeze_attn = mode == 'freeze'
            acc = {s: [0.0, 0.0] for s in sites}
            for r in range(reps):
                est.generator.manual_seed(1000 * N + r)
                for g in groups:
                    errs = est.direct_errors(g, N, 0.2)
                    for s in g:
                        G = est.reassemble(s, errs[s])
                        acc[s][0] += F.cosine_similarity(G.flatten(), G_true[s].flatten(), 0).item() / reps
                        acc[s][1] += ((G - G_true[s]).square().sum() / G_true[s].square().sum()).item() / reps
            for s, (c, e) in acc.items():
                site = s.replace('transformer.', '')
                record[f'{mode}/{site}/N{N}/cos'] = c
                record[f'{mode}/{site}/N{N}/relerr'] = e
                print(f'N={N:5d} {mode:6s} {site:22s} cos {c:6.3f}  rel err {e:9.3f}', flush=True)
with MLSweepLogger() as logger:
    logger.log(record, step=1)
