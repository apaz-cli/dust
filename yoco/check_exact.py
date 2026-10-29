"""Check the exact (non-ZO) parts of DustX against autograd, in fp32, on a random-init model.

1. exact_head_error == dL/d(raw lm_head output)
2. vjp_self_layer / vjp_cross, fed the TRUE o@ gradients, reproduce autograd's weight gradients
3. YOCO cross-decoder locality: dL_u/dY_t == 0 for u != t at every cross-decoder site
4. handoff: probe-path gradient + cache pullback == autograd dL/dX
"""
from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, recipe  # noqa: E402
from yoco.dustx import DustX  # noqa: E402
from yoco.model_yoco import build  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device('cuda')
cos = lambda a, b: F.cosine_similarity(a.flatten().double(), b.flatten().double(), 0).item()
rel = lambda a, b: ((a - b).norm() / b.norm().clamp_min(1e-30)).item()

data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:2].to(dev)
x, y = data[:, :128].contiguous(), data[:, 1:129].contiguous()

for arch in ('gpt', 'yoco'):
    cfg = GPTConfig(sequence_len=128, n_layer=4, n_head=4, n_kv_head=4, n_embd=256)
    model = build(arch, cfg, dev, 0)
    # train-like weights: perturb gates so value embeddings matter
    with torch.no_grad():
        for n_, p in model.named_parameters():
            if 've_gate' in n_:
                p.normal_(0, 0.5)
    model.transformer.wte.float()
    for ve in model.value_embeds.values():
        ve.float()
    s = recipe('10M', 256)
    est = DustX(model, s, 0)
    est.autocast = nullcontext
    est.dtype = torch.float32
    est.capture(x, y)

    # autograd truth
    model.requires_grad_(True)
    o_in = {}
    hs = []
    for i in range(cfg.n_layer):
        m = model.transformer.h[i].attn.c_proj
        def pre(mod, a, i=i):
            a[0].retain_grad()
            o_in[i] = a[0]
        hs.append(m.register_forward_pre_hook(pre))
    if arch == 'yoco':
        Xh = {}
        def hk(mod, a, out):
            out = out * 1.0
            out.retain_grad()
            Xh['res'] = out
            return out
        hs.append(model.transformer.handoff.register_forward_hook(hk))
    model.zero_grad()
    losses = model(x, y, loss_reduction='none').view_as(y)
    losses.sum().backward()
    for h in hs:
        h.remove()
    true_o = {i: o_in[i].grad.reshape(2, 128, -1) for i in o_in}
    P = dict(model.named_parameters())

    # 1. head
    head = est.reassemble('lm_head', est.exact_head_error())
    print(arch, 'head      rel', f"{rel(head, P['lm_head.weight'].grad):.2e}")

    # 2. local VJPs
    with torch.no_grad():
        for i in range(est.n_self):
            for site, err in est.vjp_self_layer(i, true_o[i]).items():
                g = est.reassemble(site, err)
                print(arch, f'vjp {site:28s} rel', f'{rel(g, P[site + ".weight"].grad):.2e}')
        if arch == 'yoco':
            res, dX = est.vjp_cross(true_o, want_handoff=True)
            for site, err in res.items():
                g = est.reassemble(site, err)
                print(arch, f'vjp {site:28s} rel', f'{rel(g, P[site + ".weight"].grad):.2e}')
            # 4. handoff: residual path (autograd through the handoff identity) + cache pullback
            Xleaf = est.inputs['transformer.handoff']
            total = Xh['res'].grad + dX
            # true dL/dX: autograd on a fresh graph from X
            print(arch, 'cache path share of |dL/dX|^2', f'{dX.square().sum().item() / total.square().sum().item():.3f}')

    if arch == 'yoco':
        # dL/dX truth: rerun with X as a leaf above the cache
        X = est.inputs['transformer.handoff'].clone().requires_grad_()
        def fwd_from_X(X):
            T = X.shape[1]
            cs = (model.cos[:, :T], model.sin[:, :T])
            k, v = model.cache(X, cs)
            x0 = F.rms_norm(model.transformer.wte(x), (cfg.n_embd,))
            h = X
            for i in range(model.n_self, cfg.n_layer):
                h = model.resid_lambdas[i] * h + model.x0_lambdas[i] * x0
                h = model.transformer.h[i](h, k, v, cs, model.window_sizes[i])
            h = F.rms_norm(h, (cfg.n_embd,))
            lg = model.lm_head(h)[..., :cfg.vocab_size].float()
            lg = 15 * torch.tanh(lg / 15)
            return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction='none').view_as(y)
        Ls = fwd_from_X(X)
        gX, = torch.autograd.grad(Ls.sum(), X)
        print(arch, 'handoff g_res+pullback vs dL/dX  rel', f'{rel(total, gX):.2e}')
        # 3. locality: cross-decoder writer at token t, check d L_u / d Y_t for u != t
        for i in range(model.n_self, cfg.n_layer):
            Yt = {}
            def oh(mod, a, out):
                t = out.detach().requires_grad_()
                Yt['y'] = t
                return t
            hh = model.transformer.h[i].mlp.c_fc.register_forward_hook(oh)
            Ls = fwd_from_X(X.detach())
            hh.remove()
            u = 100
            g, = torch.autograd.grad(Ls[0, u], Yt['y'])
            off = g[0].square().sum(-1)
            print(arch, f'cross block {i} c_fc: |dL_100/dY_t|^2 at t!=100:', f'{(off.sum() - off[u]).item():.3e}',
                  ' own:', f'{off[u].item():.3e}')
    est.close()
