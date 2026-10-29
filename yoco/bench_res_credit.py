"""Total error of DUST's direct estimate against the TRUE weight gradient, with residual credit res_gamma
for early-block sites (fp32 rewards, stock draw counts). Error = E||G - G_true||^2 / ||G_true||^2, mean cos."""
from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import recipe  # noqa: E402
from yoco.bench_local_zo import load  # noqa: E402
from yoco.dustx import DustX  # noqa: E402

run, R = sys.argv[1], int(sys.argv[2])
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')
model = load(run)
model.transformer.wte.float()
for ve in model.value_embeds.values():
    ve.float()
data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
sites = {'transformer.h.0.attn.c_proj': 38, 'transformer.h.0.mlp.c_fc': 38, 'transformer.h.1.attn.c_proj': 32,
         'transformer.h.1.mlp.c_fc': 32, 'transformer.wte': 10}
mods = dict(model.named_modules())
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
model.requires_grad_(False)
true = {s: grab[s].grad.float() for s in sites}
for rg in (0.0, 0.5, 0.8, 0.9, 0.95):
    est = DustX(model, recipe('10M', 256), res_gamma=rg)
    est.autocast = nullcontext
    row = []
    with torch.no_grad():
        est.capture(x, y)
        for s, N in sites.items():
            Gt = est.reassemble(s, true[s])
            err, cs = 0, 0
            for _ in range(R):
                G = est.reassemble(s, est.direct_errors([s], N, 0.2)[s])
                err += (G - Gt).square().sum().item() / Gt.square().sum().item() / R
                cs += F.cosine_similarity(G.flatten(), Gt.flatten(), 0).item() / R
            row.append(f'{s.replace("transformer.", "")} err {err:7.2f} cos {cs:.3f}')
    est.close()
    print(f'res_gamma {rg}: ' + ' | '.join(row), flush=True)
