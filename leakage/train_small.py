"""Train a small GPT on TinyStories with the unmodified DUST estimator, saving checkpoints for measure.py.

The model is scaled down (default 4 blocks, width 256, 512 tokens, head dim 64); the population-256
recipe is the repo's, with its per-block draw lists truncated to the first n_layer entries (n_layer <= 8).
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
from dust import DUST, GPTConfig, make_model, make_optimizer, recipe  # noqa: E402
from leakage.measure import Probe, measure_sequence, summarize  # noqa: E402



@torch.no_grad()
def evaluate(model, rows, batch_size=32):
    total = 0.0
    for batch in rows.split(batch_size):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            total += model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous()).item() * len(batch)
    return total / len(rows)


def scale_population(settings, f):
    """Scale every draw count by f, rounding to a multiple of the draw chunk (min one chunk) so that
    per-chunk reward centering still applies. Head draws must also visit all 16 vocabulary slabs equally."""
    c = settings.nrep

    def r(k, unit=c):
        return max(unit, unit * round(k * f / unit))

    return replace(settings, writers=[r(k) for k in settings.writers], hidden=[r(k) for k in settings.hidden],
                   hubs=[r(k) for k in settings.hubs], embedding=r(settings.embedding),
                   head=r(settings.head, c * 16), local={k: r(v) for k, v in settings.local.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('data/tinystories'))
    p.add_argument('--output', type=Path, default=Path('runs/leakage-small'))
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--save-at', type=int, nargs='+', default=[0, 1000, 3000])
    p.add_argument('--lr', type=float)
    p.add_argument('--population', type=int, default=256,
                   help='scales the 256 recipe (n_layer=8); below 256 counts round to the draw chunk, min one chunk')
    p.add_argument('--n-layer', type=int, default=4)
    p.add_argument('--n-embd', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--diverge-nats', type=float, default=1.0,
                   help='stop when the smoothed train loss rises this far above its minimum')
    p.add_argument('--max-loss', type=float, default=10.0, help='stop when a step\'s train loss exceeds this')
    p.add_argument('--leak-every', type=int, default=0,
                   help='measure leakage on the training batch every N steps (0 = never)')
    p.add_argument('--leak-seqs', type=int, default=2, help='sequences of the batch to measure')
    p.add_argument('--eta', action='store_true',
                   help='also log curvature ratio eta, from the actual DUST update rewards')
    args = p.parse_args()
    if args.eta and not args.leak_every:
        p.error('--eta requires --leak-every')

    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')
    heads = args.n_embd // 64
    config = GPTConfig(sequence_len=512, n_layer=args.n_layer, n_head=heads, n_kv_head=heads, n_embd=args.n_embd)
    settings = recipe('10M', 256)
    n = config.n_layer
    settings = replace(settings, writers=settings.writers[:n], hidden=settings.hidden[:n], hubs=settings.hubs[:n],
                       **({'lr': args.lr} if args.lr else {}))
    if args.population != 256:
        settings = scale_population(settings, args.population / 256)
    train = torch.load(args.data_dir / 'train.pt', weights_only=True).long()
    val = torch.load(args.data_dir / 'validation.pt', weights_only=True).long().to(device)
    if len(train) < args.steps * 8:
        raise ValueError('not enough training rows')
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + 1))
    train = train[order]

    model = make_model(config, device, args.seed)
    optimizer = make_optimizer(model, settings)
    estimator = DUST(model, settings, args.seed)
    if args.eta:
        estimator.reward_stats = {}
    probe = Probe(model) if args.leak_every else None
    leak_gen = torch.Generator().manual_seed(args.seed + 2)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(
        dict(model=asdict(config), recipe=asdict(settings), population=args.population,
             direct_draws=sum(settings.writers + settings.hidden + settings.hubs) + settings.embedding,
             steps=args.steps, seed=args.seed,
             data=str(args.data_dir)), indent=1) + '\n')

    def save(step):
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.output / f'step{step}.pt')

    log = (args.output / 'metrics.jsonl').open('w')
    leak_log = (args.output / 'leakage.jsonl').open('w') if probe else None
    eta_log = (args.output / 'eta.jsonl').open('w') if args.eta else None

    def measure_leakage(step, batch):
        # Same weights and batch the DUST update at this step sees; every target token is measured.
        per_seq = [measure_sequence(model, probe, seq, config.sequence_len, 16, leak_gen)
                   for seq in batch[:args.leak_seqs]]
        summary = summarize(per_seq, probe.sites)
        leak_log.write(json.dumps(dict(step=step, sites=summary)) + '\n')
        leak_log.flush()
        return summary

    def predicted_mean(summary, key, tokens):
        if len(key) == 1:
            s = summary[key[0]]
            return s['own'] * (1 + s['kappa']) / tokens
        # The only multi-site direct-error group is the writer pair.  kappa_group on either member is
        # already the pair's total leakage, so own_a * (1 + kappa_group_a) is the pair's predicted sum.
        s = summary[key[0]]
        return s['own'] * (1 + s['kappa_group']) / tokens

    def eta_row(step, summary, stats, tokens, nrep):
        rows = {}
        # direct_errors centers each chunk of nrep draws, which removes 1/nrep of the reward variance.
        center = nrep / (nrep - 1) if nrep > 1 else 1.0
        for key, (sum_sq, count, sigma) in stats.items():
            if count == 0:
                continue
            pred = predicted_mean(summary, key, tokens)
            if pred <= 0:
                continue
            measured = center * sum_sq.item() / count
            name = '+'.join(key)
            rows[name] = dict(eta=measured / pred - 1, measured=measured, predicted=pred, sigma=sigma)
        return dict(step=step, sites=rows)
    start = time.monotonic()
    if 0 in args.save_at:
        save(0)
    ema = best = first = None
    last_summary = None
    for step in range(1, args.steps + 1):
        batch = train[(step - 1) * 8:step * 8].to(device)
        if probe and (step == 1 or step % args.leak_every == 0):
            summary = measure_leakage(step, batch)
            if args.eta:
                # Match the rewards from the interval to the model at its start, not its end.
                if last_summary is not None and estimator.reward_stats:
                    eta_log.write(json.dumps(eta_row(step, summary, estimator.reward_stats,
                                                     config.sequence_len * args.leak_seqs, settings.nrep)) + '\n')
                    eta_log.flush()
                last_summary = summary
                estimator.reward_stats = {}
        try:
            loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        except FloatingPointError:
            loss = math.nan
        rec = dict(step=step, train_loss=loss, seconds=round(time.monotonic() - start, 1))
        first = loss if first is None else first
        ema = loss if ema is None else 0.9 * ema + 0.1 * loss
        best = ema if best is None else min(best, ema)
        if (not math.isfinite(loss) or loss > args.max_loss or ema > best + args.diverge_nats
                or (step >= 20 and ema > first)):
            rec['diverged'] = True
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + '\n')
            break
        if step % 50 == 0 or step == 1:
            rec['val_loss'] = evaluate(model, val)
        if step in args.save_at:
            save(step)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + '\n')
        log.flush()
    estimator.close()


if __name__ == '__main__':
    main()
