"""TinyStories-repeat: mean validation loss on the first half (plain LM) and second half (copyable) of each row."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yoco.bench_local_zo import load  # noqa: E402

v = torch.load('data/tsrepeat/validation.pt', weights_only=True).long()[:64].cuda()
for run in sys.argv[1:]:
    m = load(run)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        L = torch.cat([m(b[:, :-1].contiguous(), b[:, 1:].contiguous(), loss_reduction='none').float().view(len(b), -1)
                       for b in v.split(32)])
    print(f'{run:40s} first half {L[:, :255].mean().item():.3f}  second half {L[:, 256:].mean().item():.3f}', flush=True)
