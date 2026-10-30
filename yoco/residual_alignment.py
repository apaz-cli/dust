"""How aligned are the residual-stream errors at different depths? Per-token cosine of exact own-token errors
at every writer output (attn.c_proj, mlp.c_proj) and the embedding. High alignment means the draws at one depth
carry information about every other depth's error."""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoco import expected  # noqa: E402
from yoco.bench_local_zo import load  # noqa: E402

model = load(sys.argv[1])
L = model.config.n_layer
data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
sites = []
for i in range(L):
    sites += [f'transformer.h.{i}.attn.c_proj', f'transformer.h.{i}.mlp.c_proj']
mods = dict(model.named_modules())
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
g = {s: grab[s].grad.float() for s in sites}
names = [s.replace('transformer.h.', 'b').replace('.attn.c_proj', 'A').replace('.mlp.c_proj', 'M') for s in sites]
print('per-token cos of own-token residual errors (mean over tokens):')
print('      ' + ' '.join(f'{n:>6s}' for n in names))
for a, na in zip(sites, names):
    row = [F.cosine_similarity(g[a], g[b], dim=-1).mean().item() for b in sites]
    print(f'{na:>6s}' + ' '.join(f'{v:6.3f}' for v in row))
print('norms (rms per token):', ' '.join(f'{n}={g[s].norm(dim=-1).mean().item():.2e}' for n, s in zip(names, sites)))
