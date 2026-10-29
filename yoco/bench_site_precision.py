"""Cosine of DUST's direct ZO error estimate to the exact own-token error, per site, bf16 vs fp32 (TF32)."""
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

run, draws = sys.argv[1], int(sys.argv[2])
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')
model = load(run)
data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
L = model.config.n_layer
sites = [f'transformer.h.{i}.{s}' for i in range(L) for s in ('attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')]
mods = dict(model.named_modules())

# exact own-token errors at each site output (measurement only)
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
exact = {s: grab[s].grad.float() for s in sites}
for blk in model.transformer.h:
    del blk.attn.forward
if hasattr(model, 'n_self'):
    del model.cache
model.requires_grad_(False)

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
        cs = {s: F.cosine_similarity(est.direct_errors([s], draws, 0.2)[s].flatten(), exact[s].flatten(), 0).item()
              for s in sites}
    est.close()
    for i in range(L):
        print(prec, f'block {i}: ' + ' '.join(f'{s.split(".", 3)[3]} {cs[s]:.3f}' for s in sites if f'.h.{i}.' in s), flush=True)
