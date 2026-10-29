"""Time the attention-internal estimators inside one process: stock ZO local draws vs exact local VJP."""
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace  # noqa: E402
from dust import GPTConfig, recipe  # noqa: E402
from yoco.dustx import DustX  # noqa: E402
from yoco.model_yoco import build  # noqa: E402

L, D = int(sys.argv[1]), int(sys.argv[2])
cfg = GPTConfig(sequence_len=512, n_layer=L, n_head=D // 64, n_kv_head=D // 64, n_embd=D)
s = recipe('10M', 256)
s = replace(s, writers=s.writers[:L], hidden=s.hidden[:L], hubs=s.hubs[:L])
m = build('gpt', cfg, torch.device('cuda'), 0)
est = DustX(m, s)
data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
with torch.no_grad():
    est.capture(x, y)
    targets = {i: torch.randn(8, 512, D, device='cuda') for i in range(L)}

    def zo():
        for kind, count in s.local.items():
            est.local_errors(kind, count, targets)

    def vjp():
        for i in range(L):
            est.vjp_self_layer(i, targets[i])

    def full_step():  # one clean pass + every direct-error draw, the rest of a DUST step
        est.capture(x, y)
        for i in range(L):
            p = f'transformer.h.{i}.'
            est.direct_errors([p + 'attn.c_proj', p + 'mlp.c_proj'], s.writers[i], 0.2)
            est.direct_errors([p + 'mlp.c_fc'], s.hidden[i], 0.2)
            est.direct_errors([f'o@{i}'], s.hubs[i], 0.2)

    def head_zo():
        est.head_error(s.head)

    def head_exact():
        est.exact_head_error()

    res = {'zo': [], 'vjp': [], 'direct': [], 'head_zo': [], 'head_exact': []}
    for _ in range(5):
        for k, f in (('zo', zo), ('vjp', vjp), ('direct', full_step), ('head_zo', head_zo), ('head_exact', head_exact)):
            torch.cuda.synchronize(); t = time.perf_counter(); f(); torch.cuda.synchronize()
            res[k].append(time.perf_counter() - t)
    for k, v in res.items():
        print(f'L{L}-D{D} {k:7s} median {sorted(v)[2] * 1000:8.1f} ms')
