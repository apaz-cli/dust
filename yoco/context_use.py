"""Validation loss by token position: how much a checkpoint gains from context.

Prints mean loss over position buckets and the gain = loss[1..8] - loss[256..511].
"""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig  # noqa: E402
from yoco.model_yoco import build  # noqa: E402

BUCKETS = [(0, 1), (1, 8), (8, 32), (32, 128), (128, 256), (256, 512)]


@torch.no_grad()
def profile(run, step=3000):
    cfg = json.loads((Path(run) / 'config.json').read_text())
    model = build(cfg.get('args', {}).get('arch', 'gpt'), GPTConfig(**cfg['model']), torch.device('cuda'), 0)
    model.load_state_dict(torch.load(Path(run) / f'step{step}.pt', map_location='cuda', weights_only=True))
    val = torch.load('data/tinystories/validation.pt', weights_only=True).long().cuda()
    losses = []
    for b in val.split(32):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses.append(model(b[:, :-1].contiguous(), b[:, 1:].contiguous(), loss_reduction='none')
                          .view(len(b), -1).float())
    L = torch.cat(losses).mean(0)
    out = {f'{a}-{b}': L[a:b].mean().item() for a, b in BUCKETS}
    out['gain'] = L[1:8].mean().item() - L[256:512].mean().item()
    out['mean'] = L.mean().item()
    return out


if __name__ == '__main__':
    for run in sys.argv[1:]:
        r = profile(run)
        print(f'{run:36s} ' + ' '.join(f'{k}={v:.3f}' for k, v in r.items()))
