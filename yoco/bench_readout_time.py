"""Step cost of the cross-token readout: time of every residual-stream site's draws (writers, MLP inputs, embedding)
at the recipe's draw counts, with and without the readout, alternated in one process so both see the same GPU load.
Logged keys: plain_ms, readout_ms, ratio (medians over R alternations).
usage: bench_readout_time.py RUN DATA_DIR [R]
"""
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import recipe  # noqa: E402
from yoco.bench_local_zo import load, own_o_errors  # noqa: E402
from yoco.dustx import DustX  # noqa: E402
from mlsweep.logger import MLSweepLogger  # noqa: E402  (no-op outside an mlsweep worker)

run, data_dir = sys.argv[1], sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 10
torch.manual_seed(0)
model = load(run)
L = model.config.n_layer
st = recipe('10M', 256)
data = torch.load(f'{data_dir}/train.pt', weights_only=True).long()[:8].cuda() % 8192
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
o_err = own_o_errors(model, x, y, st.gamma)
est = DustX(model, st, seed=11)
est.set_precision('fast16')
times = {False: [], True: []}
with torch.no_grad():
    est.capture(x, y)
    est.readout_targets = {i: o_err[i].reshape(o_err[i].shape[0], o_err[i].shape[1], -1) for i in o_err}
    calls = [([f'transformer.h.{i}.attn.c_proj', f'transformer.h.{i}.mlp.c_proj'], st.writers[i]) for i in range(L)] + \
            [([f'transformer.h.{i}.mlp.c_fc'], st.hidden[i]) for i in range(L)] + [(['transformer.wte'], 4 * st.embedding)]
    for r in range(R + 1):
        for on in (False, True):
            est.readout = on
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for sites, n in calls:
                est.direct_errors(sites, n, 0.2)
            torch.cuda.synchronize()
            if r:  # the first round warms up
                times[on].append(1000 * (time.perf_counter() - t0))
plain, ro = statistics.median(times[False]), statistics.median(times[True])
print(f'plain {plain:.1f} ms, readout {ro:.1f} ms, ratio {ro / plain:.3f}')
with MLSweepLogger() as logger:
    logger.log({'plain_ms': plain, 'readout_ms': ro, 'ratio': ro / plain}, step=1)
