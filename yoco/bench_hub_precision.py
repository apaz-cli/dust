"""Cosine of the ZO o@ (attention output) error estimate to the exact own-token error: bf16 vs fp32 forwards."""
from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import recipe  # noqa: E402
from yoco.bench_local_zo import load, own_o_errors  # noqa: E402
from yoco.dustx import DustX  # noqa: E402

run, draws = sys.argv[1], int(sys.argv[2])
torch.manual_seed(0)
if len(sys.argv) > 3:
    torch.set_float32_matmul_precision(sys.argv[3])
model = load(run)
data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
e = own_o_errors(model, x, y, 0.98)
L = model.config.n_layer
for prec in ('bf16', 'fp32'):
    if prec == 'fp32':
        model.transformer.wte.float()
        for ve in model.value_embeds.values():
            ve.float()
    for sig_scale in (1.0,):
        est = DustX(model, recipe('10M', 256))
        if prec == 'fp32':
            est.autocast = nullcontext
        with torch.no_grad():
            est.capture(x, y)
            out = []
            for i in range(L):
                sigma = (0.2 if i < L // 2 else 0.4) * sig_scale
                g = est.direct_errors([f'o@{i}'], draws, sigma)[f'o@{i}']
                out.append(F.cosine_similarity(g.flatten(), e[i].flatten(), 0).item())
        est.close()
        print(f'{prec} sigma x{sig_scale}: ' + ' '.join(f'o@{i} {c:.3f}' for i, c in enumerate(out)), flush=True)
