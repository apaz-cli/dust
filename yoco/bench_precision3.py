"""Direct-estimate quality (cos to exact own-token error) and draw cost for bf16 / fp16 / fp32-TF32 forwards."""
from contextlib import nullcontext
from pathlib import Path
import sys
import time

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
sites = [f'transformer.h.{i}.attn.c_proj' for i in range(L)]
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
for hk in hooks:
    hk.remove()
exact = {s: grab[s].grad.float() for s in sites}
for blk in model.transformer.h:
    del blk.attn.forward
if hasattr(model, 'n_self'):
    del model.cache
model.requires_grad_(False)
model.transformer.wte.float()
for ve in model.value_embeds.values():
    ve.float()
for prec in ('fast16', 'delta-fp8', 'delta-nvfp4'):
    est = DustX(model, recipe('10M', 256))
    est.set_precision('fast16')
    est.delta_fmt = {'delta-fp8': 'fp8', 'delta-nvfp4': 'nvfp4'}.get(prec)
    with torch.no_grad():
        est.capture(x, y)
        cs = [F.cosine_similarity(est.direct_errors([s], draws, 0.2)[s].flatten(), exact[s].flatten(), 0).item() for s in sites]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for s in sites:
            est.direct_errors([s], 32, 0.2)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / (32 * L) * 1000
    est.close()
    print(f'{prec}: ' + ' '.join(f'b{i} {c:.3f}' for i, c in enumerate(cs)) + f'   {dt:.2f} ms/draw', flush=True)
