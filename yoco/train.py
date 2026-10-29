"""Train the GPT or YOCO model on TinyStories (T=512) with DUST variants or backprop.

Mirrors ~/git/dust/leakage/train_small.py: the population-256 10M recipe truncated to n_layer blocks,
3000 steps of batch 8, validation every 100 steps on the 128-row split.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, make_optimizer, recipe  # noqa: E402
from yoco import expected  # noqa: E402
from yoco.dustx import DustX  # noqa: E402
from yoco.model_yoco import build  # noqa: E402


@torch.no_grad()
def evaluate(model, rows, batch_size=32):
    total = 0.0
    for batch in rows.split(batch_size):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            total += model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous()).item() * len(batch)
    return total / len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['gpt', 'yoco'], default='gpt')
    p.add_argument('--method', choices=['dust', 'bp', 'ed'], default='dust')
    p.add_argument('--head', choices=['zo', 'exact'], default='zo')
    p.add_argument('--local', choices=['zo', 'vjp'], default='zo')
    p.add_argument('--handoff', action='store_true')
    p.add_argument('--handoff-draws', type=int)
    p.add_argument('--gamma', type=float)
    p.add_argument('--oracle', nargs='*', default=[], help='groups given their noise-free expected update')
    p.add_argument('--n-layer', type=int, default=4)
    p.add_argument('--n-embd', type=int, default=256)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--lr', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--save-at', type=int, nargs='*', default=[])
    p.add_argument('--data-dir', type=Path, default=Path('data/tinystories'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')
    heads = args.n_embd // 64
    config = GPTConfig(sequence_len=512, n_layer=args.n_layer, n_head=heads, n_kv_head=heads, n_embd=args.n_embd)
    settings = recipe('10M', 256)
    n = config.n_layer
    settings = replace(settings, writers=settings.writers[:n], hidden=settings.hidden[:n], hubs=settings.hubs[:n],
                       lr=args.lr, **({'gamma': args.gamma} if args.gamma is not None else {}))
    train = torch.load(args.data_dir / 'train.pt', weights_only=True).long()
    val = torch.load(args.data_dir / 'validation.pt', weights_only=True).long().to(device)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + 1))
    train = train[order]

    model = build(args.arch, config, device, args.seed)
    optimizer = make_optimizer(model, settings)
    estimator = None
    if args.method == 'dust':
        estimator = DustX(model, settings, args.seed, head=args.head, local=args.local,
                          handoff=args.handoff, handoff_draws=args.handoff_draws, oracle=args.oracle)
    else:
        model.requires_grad_(True)
        if args.method == 'ed':  # noise-free DUST: backprop of the expected DUST update
            expected.install(model, settings.gamma, cross_full=args.handoff)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(dict(
        args={k: str(v) for k, v in vars(args).items()}, model=asdict(config), recipe=asdict(settings),
        n_params=sum(q.numel() for q in model.parameters())), indent=1) + '\n')

    def save(step):
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.output / f'step{step}.pt')

    if 0 in args.save_at:
        save(0)
    log = (args.output / 'metrics.jsonl').open('w')
    start = time.monotonic()
    ema = None
    for step in range(1, args.steps + 1):
        batch = train[(step - 1) * 8:step * 8].to(device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        if estimator is not None:
            try:
                loss = estimator.step(x, y, optimizer)
            except FloatingPointError:
                loss = math.nan
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                l = model(x, y)
            l.backward()
            optimizer.step()
            loss = l.item()
        rec = dict(step=step, train_loss=loss, seconds=round(time.monotonic() - start, 1))
        ema = loss if ema is None else 0.9 * ema + 0.1 * loss
        if not math.isfinite(loss) or loss > 10 or (step > 50 and ema > 9):
            rec['diverged'] = True
            log.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)
            break
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            rec['val_loss'] = evaluate(model, val)
            print(json.dumps(rec), flush=True)
        if step in args.save_at:
            save(step)
        log.write(json.dumps(rec) + '\n')
        log.flush()
    if args.steps not in args.save_at:
        save(args.steps)


if __name__ == '__main__':
    main()
