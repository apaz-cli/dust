"""Where do fp16 activations get close to overflowing (65504)? Diagnosing the non-finite estimates that stopped the
8-layer dense AdamW runs near step 1000 (fw_round2 place8_dense, best8_dense_adam).

Loads a run's saved weights (the last step), runs the clean forward under the same fp16 autocast as fast16 on a few
training batches, and prints the largest magnitude at each module output, plus at relu(c_fc)^2 (computed in fp16 there).
usage: diag_overflow.py RUN_DIR DATA_DIR [BATCHES]
"""
from pathlib import Path
import json
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoco.model_yoco import build  # noqa: E402
from model import GPTConfig  # noqa: E402

run, data_dir = Path(sys.argv[1]), sys.argv[2]
nb = int(sys.argv[3]) if len(sys.argv) > 3 else 4
cfg = json.loads((run / 'config.json').read_text())
args = cfg['args']
model = build(args['arch'], GPTConfig(**cfg['model']), torch.device('cuda'), 0, n_self=None if args['n_self'] == 'None' else int(args['n_self']))
ck = sorted(run.glob('step*.pt'))[-1]
model.load_state_dict(torch.load(ck, map_location='cuda', weights_only=True))
model.eval()
data = torch.load(f'{data_dir}/train.pt', weights_only=True).long()
step = json.loads((run / 'metrics.jsonl').read_text().strip().splitlines()[-1])['step']
B = int(args['batch_seqs'])
start = (step - 1) * B % len(data)
rows = data[start:start + B].cuda()
x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()

peak = {}


def hook(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(o) and o.is_floating_point():
            peak[name] = max(peak.get(name, 0.0), o.detach().abs().max().float().item())
        if name.endswith('mlp.c_fc'):
            sq = F.relu(o).square()  # what the MLP computes next, in the same dtype
            peak[name + ' relu^2'] = max(peak.get(name + ' relu^2', 0.0), sq.abs().max().float().item())
    return f


for n, m in model.named_modules():
    if n and not list(m.children()):
        m.register_forward_hook(hook(n))
print(f'{ck.name} of {run.name}, batch of step {step}, {B} sequences in {nb} chunks', flush=True)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    for xc, yc in zip(x.chunk(nb), y.chunk(nb)):
        model(xc, yc)
for n, v in sorted(peak.items(), key=lambda kv: -kv[1])[:25]:
    print(f'{v:12.1f}  {n}', flush=True)
print('resid_lambdas', [round(float(v), 3) for v in model.resid_lambdas], flush=True)
