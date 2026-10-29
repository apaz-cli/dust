"""Dump validation loss vs token position (log-spaced buckets) for several runs to JSON."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig  # noqa: E402
from yoco.model_yoco import build  # noqa: E402

EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


@torch.no_grad()
def curve(run):
    cfg = json.loads((Path(run) / 'config.json').read_text())
    model = build(cfg.get('args', {}).get('arch', 'gpt'), GPTConfig(**cfg['model']), torch.device('cuda'), 0)
    model.load_state_dict(torch.load(Path(run) / 'step3000.pt', map_location='cuda', weights_only=True))
    val = torch.load('data/tinystories/validation.pt', weights_only=True).long().cuda()
    L = []
    for b in val.split(32):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            L.append(model(b[:, :-1].contiguous(), b[:, 1:].contiguous(), loss_reduction='none').view(len(b), -1).float())
    L = torch.cat(L).mean(0)
    return [[(a + b - 1) / 2, L[a:b].mean().item()] for a, b in zip(EDGES, EDGES[1:])]


if __name__ == '__main__':
    out = {r: curve(r) for r in sys.argv[2:]}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    for r, c in out.items():
        print(r, ' '.join(f'{v:.3f}' for _, v in c))
