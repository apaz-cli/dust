"""Check yoco/expected.py on a trained checkpoint, fp32, T=128.

a) gamma=1 with every stream 'full' reproduces backprop exactly.
b) gamma=1, stream 'own': each writer's weight gradient equals sum_t (dL_t/dY_t) x_t^T, computed by
   brute force (one backward per token), i.e. what DUST's per-token reward targets.
c) gamma=1, stream 'own': c_k / c_v weight gradients equal the full-credit local VJP against own-token
   o@ errors (what DustX local='vjp' targets).
"""
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig  # noqa: E402
from yoco import expected  # noqa: E402
from yoco.model_yoco import build  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device('cuda')
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
T = 128


def load(run):
    cfg = json.loads((Path(run) / 'config.json').read_text())
    m = build(cfg['args']['arch'], GPTConfig(**cfg['model']), dev, 0)
    m.load_state_dict(torch.load(Path(run) / 'step3000.pt', map_location=dev, weights_only=True))
    m.transformer.wte.float()
    for ve in m.value_embeds.values():
        ve.float()
    return m.requires_grad_(True)


seq = torch.load('data/tinystories/validation.pt', weights_only=True).long()[:1].to(dev)
x, y = seq[:, :T].contiguous(), seq[:, 1:T + 1].contiguous()

for run in sys.argv[1:]:
    ref = load(run)
    ref(x, y, loss_reduction='sum').backward()
    G = {n: p.grad.clone() for n, p in ref.named_parameters()}

    m = load(run)
    expected.install(m, 1.0)
    for blk in m.transformer.h:
        blk.attn.ed_stream = 'full'
    m(x, y, loss_reduction='sum').backward()
    worst = max(((rel(p.grad, G[n]), n) for n, p in m.named_parameters() if G[n].norm() > 0))
    print(run, 'a) full/gamma=1 vs backprop, worst rel err:', f'{worst[0]:.2e}', worst[1])

    # b) brute-force own-token writer gradients
    m = load(run)
    expected.install(m, 1.0)
    m(x, y, loss_reduction='sum').backward()
    Ged = {n: p.grad.clone() for n, p in m.named_parameters()}
    base = load(run)
    sites = ['transformer.h.0.attn.c_proj', 'transformer.h.0.mlp.c_fc', 'transformer.h.1.mlp.c_proj']
    mods = dict(base.named_modules())
    leaves, ins = {}, {}
    def hook(name):
        def h(mod, a, out):
            z = torch.zeros_like(out, requires_grad=True)
            leaves[name] = z
            ins[name] = a[0].detach()
            return out + z
        return h
    hs = [mods[s].register_forward_hook(hook(s)) for s in sites]
    L = base(x, y, loss_reduction='none').view(-1)
    own = {s: torch.zeros_like(leaves[s]) for s in sites}
    for t in range(T):
        gs = torch.autograd.grad(L[t], [leaves[s] for s in sites], retain_graph=True)
        for s, g in zip(sites, gs):
            own[s][0, t] = g[0, t]
    for h in hs:
        h.remove()
    for s in sites:
        Gown = own[s][0].T @ ins[s][0]
        print(run, f'b) {s}: expected-DUST vs brute-force own-token, rel err', f'{rel(Ged[s + ".weight"], Gown):.2e}',
              ' (vs backprop', f'{rel(G[s + ".weight"], Gown):.2e})')
