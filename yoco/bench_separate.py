"""Noise of the local k/v/gate/ve estimators: stock Dust (all tokens jittered together) vs leak-free separate scoring.

Targets are the exact own-token o@ errors (ruler only), so only the local estimator's own noise is measured. Both
estimators have the same expectation; the reference is the separate estimator averaged over many draws (checked
against the stock estimator at the same large count). Reported per site and draw count: relative error
||est - ref||^2 / ||ref||^2 in weight space (error rows reassembled with the site's input). The ratio stock/separate
is the measured variance cut, to compare with 1 + kappa_local.
usage: bench_separate.py RUN [REF_DRAWS] [cache]   (cache: the YOCO cache k/v, scored by every cross layer)
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import recipe  # noqa: E402
from yoco.bench_local_zo import load, own_o_errors  # noqa: E402
from yoco.dustx import DustX  # noqa: E402

run = sys.argv[1]
ref_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
cache = len(sys.argv) > 3 and sys.argv[3] == 'cache'
torch.manual_seed(0)
model = load(run)
st = recipe('10M', 256)
data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
o_err = own_o_errors(model, x, y, st.gamma)
est = DustX(model, st, seed=3)
est.set_precision('fast16')
with torch.no_grad():
    est.capture(x, y)
    targets = {i: o_err[i].reshape(o_err[i].shape[0], o_err[i].shape[1], -1) for i in o_err}

    def run_est(kind, draws, sep, seed):
        est.local_separate = est.cache_separate = sep
        est.generator.manual_seed(seed)
        return est.zo_cross(kind, draws, targets) if cache else est.zo_self_local(kind, draws, targets)

    for kind in (('k', 'v') if cache else ('k', 'v', 'gate', 've')):
        ref = run_est(kind, ref_draws, True, 1)
        chk = run_est(kind, ref_draws, False, 2)
        G = {s: est.reassemble(s, e) for s, e in ref.items()}
        rel = lambda e, s: ((est.reassemble(s, e) - G[s]).square().sum() / G[s].square().sum()).item()
        for s in ref:
            print(f'{kind:4s} {s:30s} ref check (stock @{ref_draws}) rel err {rel(chk[s], s):.3f}', flush=True)
        for N in (8, 32, 128):
            reps = 4
            acc = {s: [0.0, 0.0] for s in ref}
            for r in range(reps):
                for j, sep in enumerate((False, True)):
                    e = run_est(kind, N, sep, 100 + r)
                    for s in ref:
                        acc[s][j] += rel(e[s], s) / reps
            for s, (a, c) in acc.items():
                print(f'{kind:4s} {s:30s} N={N:4d} stock {a:8.3f} separate {c:8.3f}  cut {a / max(c, 1e-12):6.1f}x',
                      flush=True)
