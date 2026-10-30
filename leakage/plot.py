"""Plot DUST's interference ratio kappa during training.

--by-block (used in the article): one figure per run, one panel per block. Every site's kappa is its leakage
(measure.py's kappa); the attention and MLP output projections, which DUST jitters in one pass with one shared
reward, are also drawn pooled as one layer of twice the width.
Default: one panel per run, each kind of site averaged over blocks, with kappa_group (each projection's
leakage plus the other projection's noise).
--loss: every run's validation loss on one axis."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

BG, INK, MUTED, GRID = '#222222', '#fafafa', '#bbbbbb', '#3a3a3a'
RUN_COLORS = ['#3987e5', '#d95926', '#199e70']  # validated categorical slots on #222 (all-pairs CVD)


SITE_KINDS = {  # label -> site name pattern ({i} = block). Order passes the palette validator's adjacent-pair
    # CVD and normal-vision checks on #222 (leakage/plot.py legend order follows it).
    'mlp down_proj': ('transformer.h.{i}.mlp.c_proj', '#d95926'),
    'attn w_o': ('transformer.h.{i}.attn.c_proj', '#3987e5'),
    'attn o': ('o@{i}', '#d55181'),
    'embedding': ('transformer.wte', '#c98500'),
    'mlp up_proj': ('transformer.h.{i}.mlp.c_fc', '#199e70'),
}
PAIR_COLOR = '#d9d9d9'  # neutral: the pair combines the two projections above it, not a new kind of layer

ETA_KINDS = {  # label -> (site name pattern, colour).  η is only logged for these direct-error groups.
    '"writer" pair': (lambda i: f'transformer.h.{i}.attn.c_proj+transformer.h.{i}.mlp.c_proj', PAIR_COLOR),
    'mlp up_proj': (lambda i: f'transformer.h.{i}.mlp.c_fc', '#199e70'),
    'attn o': (lambda i: f'o@{i}', '#d55181'),
    'embedding': (lambda i: 'transformer.wte', '#c98500'),
}


def load(run):
    rows = [json.loads(line) for line in (Path(run) / 'leakage.jsonl').open()]
    cfg = json.loads((Path(run) / 'config.json').read_text())
    n_layer = cfg['model']['n_layer']
    steps = np.array([r['step'] for r in rows])
    series = {}
    for label, (pattern, _) in SITE_KINDS.items():
        names = sorted({pattern.format(i=i) for i in range(n_layer)})
        series[label] = np.array([np.mean([r['sites'][n]['kappa_group'] for n in names]) for r in rows])
    val = {}
    for line in (Path(run) / 'metrics.jsonl').open():
        r = json.loads(line)
        if 'val_loss' in r:
            val[r['step']] = r['val_loss']
    return steps, series, val, n_layer, cfg


def draws(cfg):
    r = cfg['recipe']
    return sum(r['writers'] + r['hidden'] + r['hubs']) + r['embedding']


def smooth(y, k):
    if k <= 1:
        return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode='edge')
    return np.convolve(pad, np.ones(k) / k, mode='valid')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', nargs='+', required=True)
    p.add_argument('--label', nargs='+')
    p.add_argument('--smooth', type=int, default=5)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--loss', action='store_true', help='plot validation loss curves instead of kappa')
    p.add_argument('--eta', action='store_true', help='plot curvature ratio eta instead of kappa')
    p.add_argument('--penalty', action='store_true', help='plot the variance penalty 1 + kappa instead of kappa')
    p.add_argument('--by-block', action='store_true',
                   help='one panel per block for a single --run (embedding drawn in block 0)')
    args = p.parse_args()
    labels = args.label or [None] * len(args.run)

    plt.rcParams.update({'font.family': 'monospace', 'font.size': 10, 'text.color': INK,
                         'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.edgecolor': GRID, 'axes.facecolor': BG, 'figure.facecolor': BG})
    if args.loss:
        return plot_loss(args, labels)
    if args.eta:
        if args.by_block:
            return plot_eta_by_block(args)
        return plot_eta(args, labels)
    if args.by_block:
        return plot_by_block(args)
    fig, axes = plt.subplots(1, len(args.run), figsize=(5.0 * len(args.run), 3.8), sharey=True, squeeze=False)
    fig.suptitle('Interference ratio κ', x=0.01, ha='left', fontsize=12, color=INK)
    ends = []  # (ax, final y, final kappa, color), labelled after layout
    for ax, run, label in zip(axes[0], args.run, labels):
        steps, series, val, n_layer, cfg = load(run)
        for name, y in series.items():
            c = SITE_KINDS[name][1]
            k = smooth(y, args.smooth)
            ys = 1 + k if args.penalty else np.maximum(k, 1e-12)
            ax.plot(steps, ys, color=c, lw=2, solid_capstyle='round', label=name)
            ends.append((ax, ys[-1], k[-1], c, ys.max()))
        last = max(val)
        model = cfg['model']
        title = label or f"{model['n_layer']}×{model['n_embd']}, {draws(cfg)} draws/step"
        ax.set_title(f'{title}  (val {val[last]:.2f})', fontsize=10, color=INK, loc='left')
        ax.set_xlabel('training step')
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_xlim(0, steps[-1])
        for sp in ax.spines.values():
            sp.set_visible(False)
    if args.penalty:
        top = max(e[4] for e in ends)
        for ax in axes[0]:
            ax.set_ylim(0.9, top * 1.08)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}×'))
        axes[0][0].set_ylabel('variance penalty 1 + κ')
    else:
        for ax in axes[0]:
            ax.set_yscale('log')
            # plain numbers (0.001, 1, 10) instead of 10^k, so 1 reads as 1
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[0][0].set_ylabel('interference ratio κ')
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=len(names), frameon=False, labelcolor=INK, fontsize=9,
               handlelength=1.5, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.subplots_adjust(right=0.92, wspace=0.26)
    # End labels: final kappa, nudged apart vertically so they don't overlap.
    fig.canvas.draw()
    for ax in axes[0]:
        mine = sorted((e for e in ends if e[0] is ax), key=lambda e: e[1])
        px = [ax.transData.transform((0, e[1]))[1] for e in mine]
        for i in range(1, len(px)):
            px[i] = max(px[i], px[i - 1] + 11)
        x_end = ax.get_xlim()[1]
        for e, p_y in zip(mine, px):
            y = ax.transData.inverted().transform((0, p_y))[1]
            ax.annotate('■', (x_end, y), xytext=(4, 0), textcoords='offset points', va='center',
                        fontsize=7, color=e[3], annotation_clip=False)
            ax.annotate(f'κ={e[2]:.2g}', (x_end, y), xytext=(13, 0), textcoords='offset points', va='center',
                        fontsize=8, color=MUTED, annotation_clip=False)
    fig.savefig(args.output, facecolor=BG)
    print(args.output)


def label_ends(ax, ends, symbol='κ'):
    """Final-value labels at the right edge, nudged apart vertically; ends = [(final y, value, color)]."""
    ends = sorted(ends)
    px = [ax.transData.transform((0, y))[1] for y, _, _ in ends]
    for i in range(1, len(px)):
        px[i] = max(px[i], px[i - 1] + 11)
    x_end = ax.get_xlim()[1]
    for (_, k, c), p_y in zip(ends, px):
        y = ax.transData.inverted().transform((0, p_y))[1]
        ax.annotate('■', (x_end, y), xytext=(4, 0), textcoords='offset points', va='center',
                    fontsize=7, color=c, annotation_clip=False)
        ax.annotate(f'{symbol}={k:.2g}', (x_end, y), xytext=(13, 0), textcoords='offset points', va='center',
                    fontsize=8, color=MUTED, annotation_clip=False)


def plot_by_block(args):
    run = Path(args.run[0])
    rows = [json.loads(line) for line in (run / 'leakage.jsonl').open()]
    cfg = json.loads((run / 'config.json').read_text())
    n = cfg['model']['n_layer']
    steps = np.array([r['step'] for r in rows])
    cols = min(n, 2)
    rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(4.3 * cols + 0.95, 2.9 * rows_ + 0.95), sharey=True,
                             sharex=True, squeeze=False)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ends, zero = [], []
        series = []
        for name, (pattern, c) in SITE_KINDS.items():
            if pattern == 'transformer.wte' and i:
                continue  # the embedding sits before block 0
            series.append((name, c, np.array([r['sites'][pattern.format(i=i)]['kappa'] for r in rows])))
        # The writer pair as one layer of twice the width: pooled leakage (leak_a + leak_m) / (own_a + own_m).
        # measure.py logs per-site ratios; the paired ratio kappa_group_a = kappa_a + own_m (1 + kappa_m) / own_a
        # recovers own_m / own_a exactly.
        a = [r['sites'][f'transformer.h.{i}.attn.c_proj'] for r in rows]
        m = [r['sites'][f'transformer.h.{i}.mlp.c_proj'] for r in rows]
        ka, km = np.array([x['kappa'] for x in a]), np.array([x['kappa'] for x in m])
        ratio = (np.array([x['kappa_group'] for x in a]) - ka) / (1 + km)
        series.insert(2, ('"writer" pair as one layer', PAIR_COLOR, (ka + km * ratio) / (1 + ratio)))
        for name, c, k in series:
            k = smooth(k, args.smooth)
            if k.max() == 0:  # no attention after the last block
                zero.append(name)
                continue
            ax.plot(steps, np.maximum(k, 1e-12), color=c, lw=2, solid_capstyle='round', label=name)
            ends.append((max(k[-1], 1e-12), k[-1], c))
        ax.set_yscale('log')
        ax.set_title(f'block {i}' + (' (+ embedding)' if i == 0 else ''), fontsize=10, color=INK, loc='left')
        if zero:
            ax.text(0.03, 0.04, 'every line: κ = 0\n(no attention after this block)', transform=ax.transAxes,
                    fontsize=8, color=MUTED)
        if i >= n - cols:
            ax.set_xlabel('training step')
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_xlim(0, steps[-1])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax._ends = ends
    for ax in axes:
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for i in range(0, n, cols):
        axes[i].set_ylabel('interference ratio κ')
    model = cfg['model']
    fig.suptitle(f"Interference ratio κ — {model['n_layer']}×{model['n_embd']}, {draws(cfg)} draws/step",
                 x=0.01, ha='left', fontsize=10, color=INK)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=4, frameon=False, labelcolor=INK, fontsize=9,
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.75 / (2.9 * rows_ + 0.95), 1, 1 - 0.25 / (2.9 * rows_ + 0.95)))
    fig.subplots_adjust(right=1 - 0.95 / (4.3 * cols + 0.95), wspace=0.38)
    fig.canvas.draw()
    for ax in axes:
        label_ends(ax, ax._ends)
    fig.savefig(args.output, facecolor=BG)
    print(args.output)


def smooth_nan(y, k):
    """Smooth a series that may contain NaNs, interpolating gaps first."""
    y = np.asarray(y, dtype=float)
    if k <= 1:
        return y
    idx = np.arange(len(y))
    good = np.isfinite(y)
    if good.sum() == 0:
        return y
    if not good.all():
        y = np.interp(idx, idx[good], y[good])
    return smooth(y, k)


def eta_series(rows, label, fn, n_layer):
    """Return eta versus step for one kind of site, averaged over blocks (embedding once)."""
    names = [fn(0)] if label == 'embedding' else [fn(i) for i in range(n_layer)]
    out = []
    for r in rows:
        vals = [r['sites'][name]['eta'] for name in names
                if name in r['sites'] and np.isfinite(r['sites'][name]['eta'])]
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


def plot_eta(args, labels):
    fig, axes = plt.subplots(1, len(args.run), figsize=(5.0 * len(args.run), 3.8), sharey=True, squeeze=False)
    fig.suptitle('Curvature ratio η', x=0.01, ha='left', fontsize=12, color=INK)
    ends = []
    for ax, run, label in zip(axes[0], args.run, labels):
        rows = [json.loads(line) for line in (Path(run) / 'eta.jsonl').open()]
        cfg = json.loads((Path(run) / 'config.json').read_text())
        n_layer = cfg['model']['n_layer']
        steps = np.array([r['step'] for r in rows])
        for name, (fn, c) in ETA_KINDS.items():
            y = smooth_nan(eta_series(rows, name, fn, n_layer), args.smooth)
            ys = np.maximum(y, 1e-6)
            ax.plot(steps, ys, color=c, lw=2, solid_capstyle='round', label=name)
            ends.append((ax, ys[-1], y[-1], c, ys.max()))
        title = label or f"{cfg['model']['n_layer']}×{cfg['model']['n_embd']}"
        ax.set_title(title, fontsize=10, color=INK, loc='left')
        ax.set_xlabel('training step')
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_xlim(0, steps[-1])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes[0]:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0][0].set_ylabel('curvature ratio η')
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=len(names), frameon=False, labelcolor=INK, fontsize=9,
               handlelength=1.5, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.subplots_adjust(right=0.92, wspace=0.26)
    fig.savefig(args.output, facecolor=BG)
    print(args.output)


def plot_eta_by_block(args):
    run = Path(args.run[0])
    rows = [json.loads(line) for line in (run / 'eta.jsonl').open()]
    cfg = json.loads((run / 'config.json').read_text())
    n = cfg['model']['n_layer']
    steps = np.array([r['step'] for r in rows])
    cols = min(n, 2)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(4.3 * cols + 0.95, 2.9 * nrows + 0.95), sharey=True,
                             sharex=True, squeeze=False)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ends, zero = [], []
        for name, (fn, c) in ETA_KINDS.items():
            if name == 'embedding' and i:
                continue
            key = fn(i)
            y = np.array([r['sites'].get(key, {}).get('eta', np.nan) for r in rows])
            if not np.isfinite(y).any():
                zero.append(name)
                continue
            k = smooth_nan(y, args.smooth)
            ys = np.maximum(k, 1e-6)
            ax.plot(steps, ys, color=c, lw=2, solid_capstyle='round', label=name)
            ends.append((max(k[-1], 1e-6), k[-1], c))
        ax.set_yscale('log')
        ax.set_title(f'block {i}' + (' (+ embedding)' if i == 0 else ''), fontsize=10, color=INK, loc='left')
        if zero:
            ax.text(0.03, 0.04, 'no η measurement', transform=ax.transAxes, fontsize=8, color=MUTED)
        if i >= n - cols:
            ax.set_xlabel('training step')
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_xlim(0, steps[-1])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax._ends = ends
    for ax in axes:
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for i in range(0, n, cols):
        axes[i].set_ylabel('curvature ratio η')
    model = cfg['model']
    fig.suptitle(f"Curvature ratio η — {model['n_layer']}×{model['n_embd']}, {draws(cfg)} draws/step",
                 x=0.01, ha='left', fontsize=10, color=INK)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=4, frameon=False, labelcolor=INK, fontsize=9,
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.75 / (2.9 * nrows + 0.95), 1, 1 - 0.25 / (2.9 * nrows + 0.95)))
    fig.subplots_adjust(right=1 - 0.95 / (4.3 * cols + 0.95), wspace=0.38)
    fig.canvas.draw()
    for ax in axes:
        label_ends(ax, ax._ends, symbol='η')
    fig.savefig(args.output, facecolor=BG)
    print(args.output)


def plot_loss(args, labels):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.set_title('Validation loss', loc='left', fontsize=12, color=INK)
    for run, label, c in zip(args.run, labels, RUN_COLORS):
        _, _, val, _, cfg = load(run)
        model = cfg['model']
        name = label or f"{model['n_layer']}×{model['n_embd']}, lr {cfg['recipe']['lr']:g}"
        steps = sorted(val)
        ys = [val[k] for k in steps]
        ax.plot(steps, ys, color=c, lw=2, label=f'{name}  (final {ys[-1]:.2f})', solid_capstyle='round')
    ax.set_xlabel('training step')
    ax.set_ylabel('validation loss')
    ax.set_ylim(top=min(4.5, ax.get_ylim()[1]))
    ax.grid(True, color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=BG)
    print(args.output)


if __name__ == '__main__':
    main()
