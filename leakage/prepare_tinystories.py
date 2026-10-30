"""Pack TinyStories (pinned revision) into [N, seq_len + 1] token rows with the repo's BPE-4096 tokenizer."""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'


def pack(stories, tok, rows, width):
    end, tokens = tok.token_to_id('<|endoftext|>'), []
    for text in stories:
        tokens.append(end)
        tokens.extend(tok.encode(text).ids)
        if len(tokens) >= rows * width:
            break
    return torch.tensor(tokens[:rows * width], dtype=torch.int32).view(rows, width)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--train-rows', type=int, default=24000)
    p.add_argument('--val-rows', type=int, default=128)
    p.add_argument('--output', type=Path, default=Path('data/tinystories'))
    args = p.parse_args()
    tok = Tokenizer.from_file(str(Path(__file__).parents[1] / 'tokenizer' / 'tokenizer.json'))
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in (('train', args.train_rows), ('validation', args.val_rows)):
        ds = load_dataset('roneneldan/TinyStories', split=split, revision=REVISION)
        data = pack((r['text'] for r in ds), tok, rows, args.seq_len + 1)
        torch.save(data, args.output / f'{split}.pt')
        print(f'{split}: {tuple(data.shape)}', flush=True)


if __name__ == '__main__':
    main()
