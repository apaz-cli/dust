"""TinyStories-repeat: each row is 256 story tokens followed by the same 256 tokens (+1 for the shift).
The second half is predictable from context by copying (an induction-style circuit), so a model that learns to
use context shows a large loss gap between halves, and its early layers carry cross-token credit."""
from pathlib import Path

import torch

out = Path('data/tsrepeat')
out.mkdir(parents=True, exist_ok=True)
for split in ('train', 'validation'):
    d = torch.load(f'data/tinystories/{split}.pt', weights_only=True)
    half = d[:, :256]
    rows = torch.cat([half, half, half[:, :1]], 1)
    torch.save(rows.contiguous(), out / f'{split}.pt')
print('wrote', out)
