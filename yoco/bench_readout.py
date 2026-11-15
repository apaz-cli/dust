"""Cross-token readout (from noise PR #46, on plain Dust) vs plain Dust, at equal draws.

Plain Dust credits each token's jitter with that token's own loss change; the change it causes in later tokens'
losses through their attention lands on those tokens' rewards instead (leakage for them, lost credit for it). The
readout adds that cross-token credit back per draw: for every later reader s of token t's key and value (lower-half
self attention, and the YOCO cache read by the upper half), the change of s's attention output from t's key and value
alone, in closed form (one changed softmax row), scored against s's attention-output (o@) error. Targets are either
the exact own-token o@ errors (best case, a ruler) or Dust's own o@ hub estimates.
Scored against the true weight gradient from autograd (a measuring stick only): cosine and relative error
||G_est - G_true||^2 / ||G_true||^2, per site, at N draws, averaged over R repeats.
Logged keys: <mode>/<site>/N<draws>/<cos|relerr>, modes plain, exact (readout, exact targets), dust (Dust targets).
With `quality`: how good the targets must be. Targets are the exact errors plus Gaussian noise scaled to a chosen
cosine c with them (mode tc<c>), Dust hubs at more draws (mode hub<draws>, their cosine logged as
hub<draws>/<i>/cos), and the chained targets training uses (mode chain: each block's c_proj error at the recipe's
draws through 256 local probes, cosine logged as chain/<i>/cos); writer draws fixed at 256. Logged keys: <mode>/<site>/N256/<cos|relerr>.
With `sym`: the symmetric readout. Readout modes credit (as above), leak (subtract the predicted leakage of earlier
tokens' keys and values into each token's own reward) and both, with exact and with the training's chained
targets, at N = 64, 256, 1024. Logged keys: <mode>_<target>/<site>/N<draws>/<cos|relerr>, plus plain.
usage: bench_readout.py RUN DATA_DIR [R] [quality|sym]
"""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import recipe  # noqa: E402
from yoco.bench_local_zo import load, own_o_errors  # noqa: E402
from yoco.dustx import DustX  # noqa: E402
from mlsweep.logger import MLSweepLogger  # noqa: E402  (no-op outside an mlsweep worker)

run, data_dir = sys.argv[1], sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 3
quality = len(sys.argv) > 4 and sys.argv[4] == 'quality'
sym = len(sys.argv) > 4 and sys.argv[4] == 'sym'
torch.manual_seed(0)
model = load(run)
L = model.config.n_layer
st = recipe('10M', 256)
data = torch.load(f'{data_dir}/train.pt', weights_only=True).long()[:8].cuda()
x, y = (data[:, :-1] % 8192).contiguous(), data[:, 1:].contiguous()
y = torch.where(y >= 8192, torch.full_like(y, -1), y)
o_err = own_o_errors(model, x, y, st.gamma)

# true gradients of the site weights (backprop, measuring stick only)
S = getattr(model, 'n_self', L)
groups = [[f'transformer.h.{i}.attn.c_proj', f'transformer.h.{i}.mlp.c_proj'] for i in range(S)] + \
         [[f'transformer.h.{i}.mlp.c_fc'] for i in range(S)] + [['transformer.wte']]  # the readout only reaches these
sites = [s for g in groups for s in g]
params = dict(model.named_parameters())
model.requires_grad_(True)
with torch.autocast('cuda', dtype=torch.bfloat16):
    model(x, y, loss_reduction='sum').backward()
G_true = {s: params[s + '.weight'].grad.float().clone() for s in sites}
model.zero_grad(set_to_none=True)
model.requires_grad_(False)

est = DustX(model, st, seed=11)
est.set_precision('fast16')
record = {}
with torch.no_grad():
    est.capture(x, y)
    exact = {i: o_err[i].reshape(o_err[i].shape[0], o_err[i].shape[1], -1) for i in o_err}
    est.generator.manual_seed(7)
    hubs = {i: est.direct_errors([f'o@{i}'], st.hubs[i], 0.2 if i < L // 2 else 0.4)[f'o@{i}'] for i in range(L)}
    for i in range(L):
        c = F.cosine_similarity(hubs[i].flatten(), exact[i].flatten(), 0).item()
        record[f'hub/{i}/cos'] = c
        print(f'o@{i} Dust hub vs exact own-token error: cos {c:.3f}', flush=True)
    def score(mode, N, reps, targets, ro_mode='credit'):
        est.readout = targets is not None
        est.readout_mode = ro_mode
        est.readout_targets = targets
        est._ro_clean = {}
        acc = {s: [0.0, 0.0] for s in sites}
        for r in range(reps):
            est.generator.manual_seed(1000 * N + r)
            for g in groups:
                errs = est.direct_errors(g, N, 0.2)
                for s in g:
                    G = est.reassemble(s, errs[s])
                    acc[s][0] += F.cosine_similarity(G.flatten(), G_true[s].flatten(), 0).item() / reps
                    acc[s][1] += ((G - G_true[s]).square().sum() / G_true[s].square().sum()).item() / reps
        for s, (c, e) in acc.items():
            site = s.replace('transformer.', '')
            record[f'{mode}/{site}/N{N}/cos'] = c
            record[f'{mode}/{site}/N{N}/relerr'] = e
            print(f'N={N:5d} {mode:8s} {site:22s} cos {c:6.3f}  rel err {e:9.3f}', flush=True)

    def chained_targets():
        est.readout = False
        est.generator.manual_seed(9)
        out = {}
        for i in range(L):
            w = est.direct_errors([f'transformer.h.{i}.attn.c_proj'], st.writers[i], 0.2)[f'transformer.h.{i}.attn.c_proj']
            out[i] = est.local_proj(i, w, 256)
        return out

    if sym:
        chain = chained_targets()
        for N in (64, 256, 1024):
            score('plain', N, R if N < 1024 else 1, None)
            for tname, targets in (('exact', exact), ('chain', chain)):
                for m in ('credit', 'leak', 'both'):
                    score(f'{m}_{tname}', N, R if N < 1024 else 1, targets, m)
    elif not quality:
        for N in (64, 256, 1024):
            for mode, targets in (('plain', None), ('exact', exact), ('dust', hubs)):
                score(mode, N, R if N < 1024 else 1, targets)
    else:
        score('plain', 256, R, None)
        g = torch.Generator(device='cuda').manual_seed(5)
        for c in (0.25, 0.5, 0.75, 0.9):
            noisy = {}
            for i, e in exact.items():  # e + z with |z| chosen so cos(e + z, e) = c (z orthogonal in expectation)
                z = torch.randn(e.shape, generator=g, device=e.device)
                noisy[i] = e + z * (e.norm() / z.norm() * (1 / c ** 2 - 1) ** 0.5)
            score(f'tc{c}', 256, R, noisy)
        for H in (64, 256, 1024):
            est.readout = False
            est.generator.manual_seed(7)
            more = {i: est.direct_errors([f'o@{i}'], H, 0.2 if i < L // 2 else 0.4)[f'o@{i}'] for i in range(L)}
            for i in range(L):
                c = F.cosine_similarity(more[i].flatten(), exact[i].flatten(), 0).item()
                record[f'hub{H}/{i}/cos'] = c
                print(f'o@{i} Dust hub at {H} draws vs exact: cos {c:.3f}', flush=True)
            score(f'hub{H}', 256, R, more)
        # the targets training uses: each block's attn.c_proj error at the recipe's draws, chained through local probes
        est.readout = False
        est.generator.manual_seed(9)
        chained = {}
        for i in range(L):
            w = est.direct_errors([f'transformer.h.{i}.attn.c_proj'], st.writers[i], 0.2)[f'transformer.h.{i}.attn.c_proj']
            chained[i] = est.local_proj(i, w, 256)
            c = F.cosine_similarity(chained[i].flatten(), exact[i].flatten(), 0).item()
            record[f'chain/{i}/cos'] = c
            print(f'o@{i} chained target ({st.writers[i]} writer draws) vs exact: cos {c:.3f}', flush=True)
        score('chain', 256, R, chained)
with MLSweepLogger() as logger:
    logger.log(record, step=1)
