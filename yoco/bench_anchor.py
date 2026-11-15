"""Anchored draws vs full truncated draws: can a site's draws stop at a nearby block instead of rerunning to the loss?

A draw at a site in block l normally reruns blocks l .. L-1 and the head, and scores each token's loss change. An
anchored draw reruns only up to block k > l (the anchor) and scores the change of the stream entering block k,
token by token, against an estimate e_k of the error there: reward_t = -e_k,t . (x_k,t' - x_k,t). Token t's own
loss depends on the stream at block k only at positions <= t, and token t's jitter reaches only positions >= t, so
with the own-token error at the anchor this is Dust's own-token target to first order (no new bias beyond the
curvature over k - l blocks). The error at the stream entering block k is the error at block k-1's mlp.c_proj
output (it is added straight into the stream), so e_k is Dust's direct estimate there, from its own draws.
Anchor errors: 'dust' (that estimate at A draws) or 'true' (the full gradient at the stream, autograd, a ruler
only: it also carries the cross-token credit above the anchor, so it is a best case, not Dust's target).
Scored against the true weight gradient (autograd, a measuring stick only): cosine, per site, at N draws, R repeats.
Cost in block passes per draw: full L - l, anchored k - l, plus the anchor's own draws, shared by every site below.
Logged keys: <mode>/<site>/N<draws>/cos, modes full, a<k>_dust<A>, a<k>_true; cost/<mode>/<site>/N<draws>.
usage: bench_anchor.py RUN DATA_DIR [R]
"""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import recipe  # noqa: E402
from yoco.bench_local_zo import load  # noqa: E402
from yoco.dustx import DustX  # noqa: E402
from mlsweep.logger import MLSweepLogger  # noqa: E402  (no-op outside an mlsweep worker)

run, data_dir = sys.argv[1], sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 3
torch.manual_seed(0)
model = load(run)
L = model.config.n_layer
st = recipe('10M', 256)
data = torch.load(f'{data_dir}/train.pt', weights_only=True).long()[:8].cuda()
x, y = (data[:, :-1] % 8192).contiguous(), data[:, 1:].contiguous()
y = torch.where(y >= 8192, torch.full_like(y, -1), y)

ANCHORS = tuple(k for k in (1, 2) if k < model.n_self)  # inside the self-decoder: the cache is not scored
groups = {l: [[f'transformer.h.{l}.attn.c_proj', f'transformer.h.{l}.mlp.c_proj'], [f'transformer.h.{l}.mlp.c_fc']]
          for l in range(max(ANCHORS))}
groups[-1] = [['transformer.wte']]  # the embedding feeds block 0
sites = [s for gs in groups.values() for g in gs for s in g]

# true gradients of the site weights and of the stream entering each anchor (backprop, measuring stick only)
params = dict(model.named_parameters())
model.requires_grad_(True)
grabbed = {}
hooks = []
for k in ANCHORS:
    def pre(mod, a, k=k):
        a[0].retain_grad()
        grabbed[k] = a[0]
    hooks.append(model.transformer.h[k].register_forward_pre_hook(pre))
with torch.autocast('cuda', dtype=torch.bfloat16):
    model(x, y, loss_reduction='sum').backward()
for h in hooks:
    h.remove()
G_true = {s: params[s + '.weight'].grad.float().clone() for s in sites}
# the hook sees the block's input after its resid_lambdas scaling; the anchor is the stream before it
e_true = {k: grabbed[k].grad.float() * model.resid_lambdas[k].float() for k in ANCHORS}
model.zero_grad(set_to_none=True)
model.requires_grad_(False)

est = DustX(model, st, seed=11)
est.set_precision('fast16')
plain_losses = est.losses
record = {}


def anchored(k, e):
    def losses(xr, yr):
        n = xr.shape[0] // est.batch
        rec = {}
        with est.autocast():
            est.model(xr, None, start=est._start, state=est.rep_state(n), record=rec)
        dx = rec['x'][k].float().view(n, est.batch, est.length, -1) - est.state['x'][k].float().unsqueeze(0)
        lin = torch.einsum('nbtd,btd->nbt', dx, e)
        return (est.clean_loss.unsqueeze(0) + lin).view(n * est.batch, est.length)
    return losses


def first_block(s):
    return int(s.split('.')[2]) if s.startswith('transformer.h.') else 0


with torch.no_grad():
    est.capture(x, y)
    modes = [('full', None, None)]
    for k in ANCHORS:
        modes.append((f'a{k}_true', k, e_true[k]))
        for A in (256, 1024):
            est.generator.manual_seed(5 + A)
            e = est.direct_errors([f'transformer.h.{k - 1}.mlp.c_proj'], A, 0.2)[f'transformer.h.{k - 1}.mlp.c_proj']
            c = F.cosine_similarity(e.flatten(), e_true[k].flatten(), 0).item()
            record[f'anchor/{k}/dust{A}/cos'] = c
            print(f'anchor {k}: Dust error at {A} draws vs true stream gradient: cos {c:.3f}', flush=True)
            modes.append((f'a{k}_dust{A}', k, e))
    for N in (64, 256, 1024):
        for mode, k, e in modes:
            est.losses = plain_losses if k is None else anchored(k, e)
            acc = {}
            for l, gs in groups.items():
                if k is not None and max(l, 0) >= k:
                    continue  # the anchor must be above the site
                for g in gs:
                    for r in range(R if N < 1024 else 1):
                        est.generator.manual_seed(1000 * N + r)
                        errs = est.direct_errors(g, N, 0.2)
                        for s in g:
                            G = est.reassemble(s, errs[s])
                            acc[s] = acc.get(s, 0.0) + F.cosine_similarity(G.flatten(), G_true[s].flatten(), 0).item() \
                                / (R if N < 1024 else 1)
            for s, c in acc.items():
                site = s.replace('transformer.', '')
                l = first_block(s)
                record[f'{mode}/{site}/N{N}/cos'] = c
                record[f'cost/{mode}/{site}/N{N}'] = N * ((L - l) if k is None else (k - l))
                print(f'N={N:5d} {mode:12s} {site:22s} cos {c:6.3f}', flush=True)
    est.losses = plain_losses
with MLSweepLogger() as logger:
    logger.log(record, step=1)
