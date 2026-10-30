"""Does KV-credit augmentation move DUST's direct estimates toward the true gradient, and at what noise cost?

Per site: weight-space cosine to the TRUE gradient with many draws (bias), and total error ||G - G_true||^2 /
||G_true||^2 at stock draw counts averaged over repeats (what one step sees). The K/V errors come from the real
derivative-free pipeline (hub + local draws), so their noise is included. fast16 rewards.
"""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace  # noqa: E402
from dust import recipe  # noqa: E402
from yoco.bench_local_zo import load  # noqa: E402
from yoco.dustx import DustX  # noqa: E402

run, R = sys.argv[1], int(sys.argv[2])
hub_scale, local_scale = int(sys.argv[3]), int(sys.argv[4])
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')
model = load(run)
model.transformer.wte.float()
for ve in model.value_embeds.values():
    ve.float()
L = model.config.n_layer
data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
sites = {'transformer.h.0.attn.c_proj': (0, 38), 'transformer.h.0.mlp.c_fc': (0, 38),
         'transformer.h.1.attn.c_proj': (1, 32), 'transformer.wte': (-1, 40)}
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
st = recipe('10M', 256)
st = replace(st, hubs=[v * hub_scale for v in st.hubs[:L]], local={k: v * local_scale for k, v in st.local.items()},
             writers=st.writers[:L], hidden=st.hidden[:L])
exact_mode = 'exact' in sys.argv
if exact_mode:
    from yoco.bench_local_zo import own_o_errors
    eo = own_o_errors(model, x, y, st.gamma)
est = DustX(model, st, seed=3)
est.set_precision('fast16')
est.kv_credit = True
est.linear_v = 'linv' in sys.argv
est.kv_credit_values_only = 'vonly' in sys.argv
cos = lambda a, b: F.cosine_similarity(a.flatten(), b.flatten(), 0).item()
with torch.no_grad():
    est.capture(x, y)
    Gt = {s: est.reassemble(s, true[s]) for s in sites}

    def pipeline():
        est.kv_strict = {}
        tg = {l: est.direct_errors([f'o@{l}'], st.hubs[l], 0.2 if l < L // 2 else 0.4)[f'o@{l}'] for l in range(L)}
        if est.yoco:
            for k in ('q', 'k', 'v'):
                est.zo_cross(k, st.local[k], tg)
        for k, c in st.local.items():
            est.zo_self_local(k, c, tg)

    if exact_mode:  # measuring stick: exact strictly-later K/V errors from autograd, own-token o@ errors
        from yoco.bench_local_zo import Reader
        exact_strict = {}
        for site in (['cache'] if est.yoco else []) + list(range(est.n_self if est.yoco else L)):
            for kind in ('k', 'v'):
                rd = Reader(est, site, kind)
                exact_strict[rd.sitename] = rd.expected(eo, st.gamma, strict=True)

        def pipeline():
            est.kv_strict = dict(exact_strict)

    for s, (layer, n_stock) in sites.items():
        res = {}
        for aug in (False, True):
            big, errs = 0, []
            for r in range(R):
                pipeline()
                est._aug = est.aug_names(layer) if aug else {}
                G = est.reassemble(s, est.direct_errors([s], n_stock, 0.2)[s])
                errs.append((G - Gt[s]).square().sum().item() / Gt[s].square().sum().item())
                big = big + G
            est._aug = {}
            res[aug] = (cos(big, Gt[s]), sum(errs) / len(errs))
        print(f'{s:30s} plain: cos(mean of {R})={res[False][0]:.3f} step err={res[False][1]:.2f} | '
              f'kv-credit: cos={res[True][0]:.3f} step err={res[True][1]:.2f}', flush=True)
