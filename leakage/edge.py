"""Find the edge of stability for one model size, then run the leakage measurement there.

--axis lr:          at population 256, double/halve the LR from --start until stable and diverged LRs
                    bracket the edge, then bisect in log space; the edge is the largest stable LR.
--axis population:  at fixed --lr, halve the population scale (--population P scales every draw count
                    by P/256; logs and plots report the resulting direct draws per step) from 256 until a run diverges (or every site is
                    at its one-chunk minimum), then bisect; the edge is the smallest stable population.

Probe runs are short (--probe-steps) and are killed early by train_small.py's divergence rules. The full
leakage run uses the edge; if it diverges later, it steps back toward stability by the bisection resolution.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def train(py, out, steps, size, lr, population, extra=()):
    def finished():
        if not (out / 'metrics.jsonl').exists():
            return False
        rows = [json.loads(line) for line in (out / 'metrics.jsonl').open()]
        return len(rows) == steps or (rows and rows[-1].get('diverged'))

    if not finished():  # also reruns a run that was interrupted
        out.parent.mkdir(parents=True, exist_ok=True)
        with (out.parent / (out.name + '.log')).open('w') as f:
            subprocess.run([py, 'leakage/train_small.py', '--lr', str(lr), '--population', str(population),
                            '--steps', str(steps), '--n-layer', str(size[0]), '--n-embd', str(size[1]),
                            '--output', str(out), *extra], stdout=f, stderr=subprocess.STDOUT, check=False)
    rows = [json.loads(line) for line in (out / 'metrics.jsonl').open()]
    stable = len(rows) == steps and not rows[-1].get('diverged')
    val = next((r['val_loss'] for r in reversed(rows) if 'val_loss' in r), None)
    r = json.loads((out / 'config.json').read_text())['recipe']
    draws = sum(r['writers'] + r['hidden'] + r['hubs']) + r['embedding']
    return stable, val, len(rows), draws


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-layer', type=int, required=True)
    p.add_argument('--n-embd', type=int, required=True)
    p.add_argument('--axis', choices=['lr', 'population'], default='population')
    p.add_argument('--lr', type=float, default=0.25, help='start LR (axis lr) or fixed LR (axis population)')
    p.add_argument('--min-population', type=int, default=16, help='below this every site is at one chunk')
    p.add_argument('--probe-steps', type=int, default=300)
    p.add_argument('--bisect', type=int, default=2)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--probe-only', action='store_true', help='skip the full leakage run')
    p.add_argument('--python', default=sys.executable)
    args = p.parse_args()
    size = (args.n_layer, args.n_embd)
    root = Path(f'runs/edge/L{size[0]}-D{size[1]}')

    if args.axis == 'lr':
        def run(x, out, steps, extra=()):
            return train(args.python, out, steps, size, x, 256, extra)
        name, x, grow = 'lr', args.lr, 2.0  # "toward the edge" = larger LR
    else:
        def run(x, out, steps, extra=()):
            return train(args.python, out, steps, size, args.lr, int(round(x)), extra)
        name, x, grow = 'population', 256, 0.5  # "toward the edge" = smaller population
        root = root / f'pop-lr{args.lr:.4g}'

    results = {}

    def probe(x):
        stable, val, n, draws = run(x, root / f'probe-{name}{x:.4g}', args.probe_steps)
        results[x] = (stable, val, draws)
        print(f'{name}={x:.4g} draws={draws} {"stable" if stable else "diverged"} steps={n} val={val}', flush=True)
        return stable

    stable = probe(x)
    step = grow if stable else 1 / grow
    while True:
        if args.axis == 'population' and x * step < args.min_population:
            break
        x *= step
        if probe(x) != results[x / step][0]:
            break
    safe = [k for k, v in results.items() if v[0]]
    risky = [k for k, v in results.items() if not v[0]]
    edge = (max if grow > 1 else min)(safe) if safe else None
    over = (min if grow > 1 else max)(risky) if risky else None
    for _ in range(args.bisect if edge and over else 0):
        mid = (edge * over) ** 0.5
        if args.axis == 'population':
            mid = round(mid)
            if mid in (edge, over):
                break
        edge, over = (mid, over) if probe(mid) else (edge, mid)
    summary = dict(n_layer=size[0], n_embd=size[1], axis=name, edge=edge, first_diverged=over,
                   probes={f'{k:.4g}': dict(stable=v[0], val=v[1], draws=v[2]) for k, v in sorted(results.items())})
    if args.axis == 'population':
        summary['lr'] = args.lr
    print(json.dumps(summary), flush=True)
    if edge is None:
        raise SystemExit('no stable setting found')
    best = min((k for k, v in results.items() if v[0]), key=lambda k: results[k][1])
    summary['best_val_probe'] = best
    (root / 'edge.json').write_text(json.dumps(summary, indent=1) + '\n')
    if args.probe_only:
        return

    back = (edge / over) if over else 1.0  # one bisection step back toward stability
    x = edge
    for _ in range(6):
        out = root / f'leakage-{name}{x:.4g}'
        stable, val, n, draws = run(x, out, args.steps, ('--leak-every', '25', '--save-at', '0', str(args.steps)))
        print(f'leakage run {name}={x:.4g} draws={draws}: {n} steps, stable={stable}, val={val}', flush=True)
        if stable:
            break
        x = x * back if back != 1.0 else x * (1 / grow)
    summary['leakage_run'], summary[f'leakage_run_{name}'] = str(out), x
    (root / 'edge.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
