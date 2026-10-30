"""Upper bound for residual pooling: combine every writer site's ZO error estimate into each site's estimate.

For site s, pooled_s = sum_j w_sj k_sj est_j with k_sj = <g_s, g_j> / ||g_j||^2 (oracle scale from exact own-token
errors) and w_sj proportional to k_sj^2-scaled inverse noise; own-only uses est_s alone. Reported: per-step error
||G - G_own||^2 / ||G_own||^2 in weight space at stock draw counts, mean over repeats. fast16 rewards.
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import recipe  # noqa: E402
from yoco import expected  # noqa: E402
from yoco.bench_local_zo import load  # noqa: E402
from yoco.dustx import DustX  # noqa: E402

run, R = sys.argv[1], int(sys.argv[2])
torch.manual_seed(0)
model = load(run)
L = model.config.n_layer
data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
st = recipe('10M', 256)
sites = [(f'transformer.h.{i}.attn.c_proj', f'transformer.h.{i}.mlp.c_proj', st.writers[i]) for i in range(L)]
flat = [s for a, b, _ in sites for s in (a, b)]
mods = dict(model.named_modules())
expected.install(model, 0.98)
model.requires_grad_(True)
grab, hooks = {}, []
for s in flat:
    def h(mod, a, out, s=s):
        out.retain_grad()
        grab[s] = out
    hooks.append(mods[s].register_forward_hook(h))
with torch.autocast('cuda', dtype=torch.bfloat16):
    model(x, y, loss_reduction='sum').backward()
for hk in hooks:
    hk.remove()
own = {s: grab[s].grad.float() for s in flat}
for blk in model.transformer.h:
    del blk.attn.forward
if hasattr(model, 'n_self'):
    del model.cache
model.requires_grad_(False)
est = DustX(model, st, seed=2)
est.set_precision('fast16')
with torch.no_grad():
    est.capture(x, y)
    G = {s: est.reassemble(s, own[s]) for s in flat}
    errs = {s: [0.0, 0.0] for s in flat}
    # per-site noise level (per element variance) from repeated estimates
    for r in range(R):
        e = {}
        for a, b, n in sites:
            e.update(est.direct_errors([a, b], n, 0.2))
        noise = {s: (e[s] - own[s]).square().mean().item() for s in flat}
        for s in flat:
            num, den = 0, 0
            for j in flat:
                k = (own[s] * own[j]).sum() / own[j].square().sum()   # oracle scale: g_s ~ k g_j
                prec = 1.0 / (k.item() ** 2 * noise[j] + 1e-30)
                num = num + prec * k * e[j]
                den = den + prec
            pooled = num / den
            for idx, est_s in enumerate((e[s], pooled)):
                Gs = est.reassemble(s, est_s)
                errs[s][idx] += ((Gs - G[s]).square().sum() / G[s].square().sum()).item() / R
    for s in flat:
        print(f'{s:30s} own-only step err {errs[s][0]:7.2f}   pooled {errs[s][1]:7.2f}   ({errs[s][0] / errs[s][1]:.1f}x)', flush=True)
