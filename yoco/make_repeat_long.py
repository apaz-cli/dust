"""Longer copy task: each row is H story tokens followed by the same H tokens (+1 for the shift), so the copied
half is T/2 = H tokens back. The story half joins the first 512 tokens of H/512 consecutive TinyStories rows.
usage: make_repeat_long.py H   (writes data/tsrepeat<2H>)"""
from pathlib import Path
import sys

import torch

H = int(sys.argv[1])
k = H // 512
out = Path(f'data/tsrepeat{2 * H}')
out.mkdir(parents=True, exist_ok=True)
for split in ('train', 'validation'):
    d = torch.load(f'data/tinystories/{split}.pt', weights_only=True)
    d = d[:len(d) // k * k, :512].reshape(-1, H)
    rows = torch.cat([d, d, d[:, :1]], 1)
    torch.save(rows.contiguous(), out / f'{split}.pt')
    print(split, tuple(rows.shape))
print('wrote', out)
