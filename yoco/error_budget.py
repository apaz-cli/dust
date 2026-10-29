"""Per-site error budget of DUST's direct estimate, in weight space, normalized by ||G_true||^2.

  bias^2     ||G_own - G_true||^2                      what the per-token reward misses (cross-token credit)
  noise      N * E||G_fp32(N) - G_own||^2              draw noise incl. leakage, per draw
  rounding   N * (E||G_bf16 - G_own||^2 - E||G_fp32 - G_own||^2)   extra variance from bf16 rewards, per draw
The error of one step with N draws is bias^2 + (noise + rounding) / N. G_true and G_own come from autograd
(measurement only); the estimates are DUST's derivative-free ones.
"""
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import recipe  # noqa: E402
from yoco import expected  # noqa: E402
from yoco.bench_local_zo import load  # noqa: E402
from yoco.dustx import DustX  # noqa: E402

run, N, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')
model = load(run)
data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
L = model.config.n_layer
sites = [f'transformer.h.{i}.{s}' for i in range(L) for s in ('attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')] + ['transformer.wte']
mods = dict(model.named_modules())


def site_grads(own):
    if own:
        expected.install(model, 0.98)
    model.requires_grad_(True)
    grab, hooks = {}, []
    for s in sites:
        def h(mod, a, out, s=s):
            out.retain_grad()
            grab[s] = out
        hooks.append(mods[s].register_forward_hook(h))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        model(x, y, loss_reduction='sum').backward()
    for hk in hooks:
        hk.remove()
    if own:
        for blk in model.transformer.h:
            del blk.attn.forward
        if hasattr(model, 'n_self'):
            del model.cache
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return {s: grab[s].grad.float() for s in sites}


true_err, own_err = site_grads(False), site_grads(True)
est = DustX(model, recipe('10M', 256))
with torch.no_grad():
    est.capture(x, y)
G = lambda s, e: est.reassemble(s, e)
Gt = {s: G(s, true_err[s]) for s in sites}
Go = {s: G(s, own_err[s]) for s in sites}
est.close()

mse = {}
for prec in ('bf16', 'fp32'):
    if prec == 'fp32':
        model.transformer.wte.float()
        for ve in model.value_embeds.values():
            ve.float()
    est = DustX(model, recipe('10M', 256))
    if prec == 'fp32':
        est.autocast = nullcontext
    with torch.no_grad():
        est.capture(x, y)
        for s in sites:
            acc = 0
            for _ in range(R):
                Z = G(s, est.direct_errors([s], N, 0.2)[s])
                acc += (Z - Go[s]).square().sum().item()
            mse[(prec, s)] = acc / R
    est.close()

out = {}
for s in sites:
    nt = Gt[s].square().sum().item()
    b2 = (Go[s] - Gt[s]).square().sum().item() / nt
    noise = N * mse[('fp32', s)] / nt
    rnd = N * (mse[('bf16', s)] - mse[('fp32', s)]) / nt
    out[s] = dict(bias2=b2, noise_per_draw=noise, rounding_per_draw=rnd)
    print(f'{s:28s} bias^2 {b2:8.4f}   noise/draw {noise:9.1f}   rounding/draw {rnd:10.1f}', flush=True)
Path(f'{run}/error_budget.json').write_text(json.dumps(out, indent=1))
