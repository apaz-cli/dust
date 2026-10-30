"""Local chaining: c_fc and o@ errors from cheap local probes against pooled stream errors, vs stock direct draws.
Per-step error vs the exact own-token target, over consecutive batches after the pooling moments warm up."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import recipe  # noqa: E402
from yoco import expected  # noqa: E402
from yoco.bench_local_zo import load  # noqa: E402
from yoco.dustx import DustX  # noqa: E402

run, steps, warm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(0)
model = load(run)
L = model.config.n_layer
st = recipe('10M', 256)
sites = [(f'transformer.h.{i}.attn.c_proj', f'transformer.h.{i}.mlp.c_proj', st.writers[i]) for i in range(L)]
flat = [s for a, b, _ in sites for s in (a, b)] + ['transformer.wte'] + [f'transformer.h.{i}.mlp.c_fc' for i in range(L)]
mods = dict(model.named_modules())
train = torch.load('data/tinystories/train.pt', weights_only=True).long()


def own_errors(x, y):
    model.lm_head.__dict__.pop('forward', None)   # undo fast16's head (no autograd rule) for the ruler
    expected.install(model, 0.98)
    model.requires_grad_(True)
    grab, hooks = {}, []
    for s in flat:
        def h(mod, a, out, s=s):
            out.retain_grad()
            grab[s] = out
        hooks.append(mods[s].register_forward_hook(h))
    for i in range(L):
        def pre(mod, a, i=i):
            a[0].retain_grad()
            grab[f'o@{i}'] = a[0]
        hooks.append(mods[f'transformer.h.{i}.attn.c_proj'].register_forward_pre_hook(pre))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        model(x, y, loss_reduction='sum').backward()
    for hk in hooks:
        hk.remove()
    for blk in model.transformer.h:
        del blk.attn.forward
    if hasattr(model, 'n_self'):
        del model.cache
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return {k: v.grad.float() for k, v in grab.items()}


est = DustX(model, st, seed=2)
est.pool = True
acc = {s: [0.0, 0.0] for s in flat + [f'o@{i}' for i in range(L)]}
count = 0
for step in range(steps):
    batch = train[step * 8:(step + 1) * 8].cuda()
    x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
    own = own_errors(x, y)
    est.set_precision('fast16')
    with torch.no_grad():
        est.capture(x, y)
        e = {}
        for a, b, n in sites:
            e.update(est.direct_errors([a, b], n, 0.2))
        e.update(est.direct_errors(['transformer.wte'], st.embedding, 0.2))
        for i in range(L):
            e.update(est.direct_errors([f'transformer.h.{i}.mlp.c_fc'], st.hidden[i], 0.2))
            e[f'o@{i}'] = est.direct_errors([f'o@{i}'], st.hubs[i], 0.2 if i < L // 2 else 0.4)[f'o@{i}']
        pooled = est.pool_writers({k: v for k, v in e.items() if not k.endswith('c_fc') and not k.startswith('o@')})
        est._halves = {}
        for i in range(L):
            pooled[f'transformer.h.{i}.mlp.c_fc'] = est.local_mlp(i, pooled[f'transformer.h.{i}.mlp.c_proj'], 256)
            pooled[f'o@{i}'] = est.local_proj(i, pooled[f'transformer.h.{i}.attn.c_proj'], 256)
        if step >= warm:
            count += 1
            for s in acc:
                if s.startswith('o@'):  # no weight here: compare the per-token errors themselves
                    G = own[s].reshape(e[s].shape)
                    for k, v in enumerate((e[s], pooled[s])):
                        acc[s][k] += ((v - G).square().sum() / G.square().sum()).item()
                    continue
                G = est.reassemble(s, own[s])
                for k, v in enumerate((e[s], pooled[s])):
                    acc[s][k] += ((est.reassemble(s, v) - G).square().sum() / G.square().sum()).item()
for s in acc:
    o, p = acc[s][0] / count, acc[s][1] / count
    print(f'{s:30s} stock {o:7.2f}   pooled/chained {p:7.2f}   ({o / p:.1f}x)', flush=True)
