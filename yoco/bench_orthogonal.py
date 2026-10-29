"""Orthogonal vs Gaussian probe directions at direct DUST sites, fp32 rewards.

Orthogonal design: each step draws a random orthogonal basis Q (d x d); draw j jitters every token along
row j of Q times sqrt(d), with an independent random sign per token. With N = d draws, a site whose per-token
loss depends only on its own token (the YOCO cross-decoder: zero leakage) is recovered exactly, up to
curvature. Where earlier tokens' jitter reaches L_t (self-decoder, any GPT block), the random signs turn that
leakage into noise that does not average out within one basis.
Reported: cosine of the per-token error estimate to the exact own-token error, vs draws.
"""
from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import recipe  # noqa: E402
from yoco import expected  # noqa: E402
from yoco.bench_local_zo import load  # noqa: E402
from yoco.dustx import DustX  # noqa: E402

run = sys.argv[1]
torch.manual_seed(0)
torch.set_float32_matmul_precision(sys.argv[3] if len(sys.argv) > 3 else 'highest')
model = load(run)
data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
L = model.config.n_layer
sites = [f'transformer.h.{i}.attn.c_proj' for i in range(L)]
mods = dict(model.named_modules())
model.transformer.wte.float()
for ve in model.value_embeds.values():
    ve.float()
expected.install(model, 0.98)
model.requires_grad_(True)
grab, hooks = {}, []
for s in sites:
    def h(mod, a, out, s=s):
        out.retain_grad()
        grab[s] = out
    hooks.append(mods[s].register_forward_hook(h))
model(x, y, loss_reduction='sum').backward()
for hk in hooks:
    hk.remove()
own = {s: grab[s].grad.float() for s in sites}
for blk in model.transformer.h:
    del blk.attn.forward
if hasattr(model, 'n_self'):
    del model.cache
model.requires_grad_(False)

est = DustX(model, recipe('10M', 256))
est.autocast = nullcontext
with torch.no_grad():
    est.capture(x, y)
    b, t = x.shape
    n = 8
    xr, yr = x.repeat(n, 1), y.repeat(n, 1)
    for s in sites:
        d = est.outputs[s].shape[-1]
        row = []
        for N in (64, 256):
            for design in ('gauss', 'orth'):
                sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
                if design == 'orth':
                    Q = torch.linalg.qr(torch.randn(d, d, device='cuda'))[0] * d ** 0.5  # rows orthogonal, norm sqrt(d)
                    perm = torch.randperm(d, device='cuda')[:N]
                    sign = torch.randint(0, 2, (b, t, 1), device='cuda').float() * 2 - 1
                errs = torch.zeros(b, t, d, device='cuda')
                for c in range(N // n):
                    if design == 'orth':
                        a = Q[perm[c * n:(c + 1) * n]].view(n, 1, 1, d) * sign.unsqueeze(0)
                    else:
                        a = torch.randn(n, b, t, d, device='cuda')
                    for sg in (1, -1):  # central difference: removes the even-order curvature terms
                        est.jitter = {s: sg * sigma * a.reshape(n * b, t, d)}
                        Lp = est.losses(xr, yr).view(n, b, t)
                        est.jitter = {}
                        errs += sg * torch.einsum('nbt,nbtd->btd', Lp, a) / (2 * sigma)
                g = errs / N
                row.append(f'{design} N={N}: {F.cosine_similarity(g.flatten(), own[s].flatten(), 0).item():.3f}')
        print(s, ' | '.join(row), flush=True)
