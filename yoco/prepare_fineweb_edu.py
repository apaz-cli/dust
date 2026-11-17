"""fineweb-edu for the YOCO runs: harder text than TinyStories, with effectively unlimited tokens.

Streams HuggingFaceFW/fineweb-edu (sample-10BT), encodes with the Dust tokenizer (4096 tokens, the models' vocabulary),
joins documents with <|endoftext|> and cuts the stream into rows of ROW tokens (the sequence length plus one, for the
shifted targets). The first VAL rows come from their own documents and form the validation set; training rows follow.
Stored as int16 (the vocabulary fits). Writes data/fineweb_edu/{train,validation}.pt and manifest.json.
usage: prepare_fineweb_edu.py [TRAIN_TOKENS] [ROW] [VAL]   (defaults 200M tokens, 513, 512)
"""
import json
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

TRAIN_TOKENS = int(float(sys.argv[1])) if len(sys.argv) > 1 else 200_000_000
ROW = int(sys.argv[2]) if len(sys.argv) > 2 else 513
VAL = int(sys.argv[3]) if len(sys.argv) > 3 else 512
out = Path(__file__).resolve().parents[1] / 'data' / 'fineweb_edu'
out.mkdir(parents=True, exist_ok=True)
tok = Tokenizer.from_file(str(Path.home() / 'git' / 'dust' / 'tokenizer' / 'tokenizer.json'))
eot = tok.token_to_id('<|endoftext|>')
docs = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True))


def rows_from(n_rows):
    """n_rows rows of ROW tokens, from whole documents (the last one cut)."""
    buf, chunks, have = [], [], 0
    while have < n_rows * ROW:
        batch = [next(docs)['text'] for _ in range(512)]
        for ids in tok.encode_batch(batch):
            buf.append(eot)
            buf.extend(ids.ids)
        if len(buf) >= 1 << 22:
            chunks.append(torch.tensor(buf, dtype=torch.int16))
            have += len(buf)
            buf = []
            print(f'{have / 1e6:.0f}M tokens', flush=True)
    if buf:
        chunks.append(torch.tensor(buf, dtype=torch.int16))
    flat = torch.cat(chunks)[:n_rows * ROW]
    return flat.view(n_rows, ROW).clone()


val = rows_from(VAL)
torch.save(val, out / 'validation.pt')
train = rows_from(TRAIN_TOKENS // ROW)
torch.save(train, out / 'train.pt')
(out / 'manifest.json').write_text(json.dumps(dict(
    corpus='HuggingFaceFW/fineweb-edu', subset='sample-10BT', tokenizer='~/git/dust/tokenizer/tokenizer.json',
    row=ROW, train_rows=len(train), validation_rows=len(val), dtype='int16'), indent=2) + '\n')
print('wrote', out, tuple(train.shape), tuple(val.shape), flush=True)
