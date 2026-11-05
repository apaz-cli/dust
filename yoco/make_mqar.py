"""Multi-query associative recall (Zoology's MQAR), a context-dependent testbed for leakage and bias.

Each 513-token row: token 0, then P key-value pairs (keys distinct, from 1..2047; values from 2048..4095), then the
P keys again in a random order, each followed by its value. Only the answers (the value after a repeated key) are
scored; every other target is stored as token + 8192, which yoco/train.py reads as "no loss". The answers can only
be predicted by retrieving from context (an induction-style circuit through attention).
"""
from pathlib import Path
import sys

import torch

NOLOSS = 8192


def make(n, pairs, seed, nk=2047, nv=2048):
    """Rows of 512 // (4 pairs) independent blocks: P pairs, then the P keys again (shuffled) with their values."""
    g = torch.Generator().manual_seed(seed)
    blocks = 512 // (4 * pairs)
    rows = torch.full((n, 513), NOLOSS, dtype=torch.int32)
    for r in range(n):
        parts, ok = [torch.zeros(1, dtype=torch.long)], [torch.zeros(1, dtype=torch.bool)]
        for _ in range(blocks):
            keys = torch.randperm(nk, generator=g)[:pairs] + 1
            vals = torch.randint(2048, 2048 + nv, (pairs,), generator=g)
            perm = torch.randperm(pairs, generator=g)
            parts += [torch.stack([keys, vals], 1).flatten(), torch.stack([keys[perm], vals[perm]], 1).flatten()]
            m = torch.zeros(4 * pairs, dtype=torch.bool)
            m[2 * pairs + 1::2] = True               # answers in the query half
            ok.append(m)
        seq, tgt_ok = torch.cat(parts), torch.cat(ok)
        seq = torch.where(tgt_ok, seq, seq + NOLOSS)
        rows[r, :len(seq)] = seq.to(torch.int32)
    return rows


if __name__ == '__main__':
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    nk = int(sys.argv[2]) if len(sys.argv) > 2 else 2047
    nv = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    out = Path(f'data/mqar{pairs}' + (f'_k{nk}v{nv}' if len(sys.argv) > 2 else ''))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(make(24000, pairs, 0, nk, nv), out / 'train.pt')
    torch.save(make(128, pairs, 1, nk, nv), out / 'validation.pt')
    print('wrote', out)
