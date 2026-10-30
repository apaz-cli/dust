"""Measure DUST's curvature ratio eta in fp32 at saved checkpoints.

eta_t = E[r_t^2] / ||grad_t L_t||^2, where r_t is the part of the finite-difference reward that is
nonlinear in the perturbation.  We measure it the same way measure.py's empirical check does:
    eta = (reward variance over DUST draws) / (first-order predicted variance) - 1
but in fp32, because bf16 loss-difference rounding is a large fresh noise source for small-variance
sites (deep blocks).  The output is appended to <run>/eta.jsonl, one row per checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import leakage.measure as measure  # noqa: E402
from dust import GPTConfig, make_model  # noqa: E402
from leakage.measure import Probe, empirical_check, per_token_losses  # noqa: E402


def predicted_vars(model, probe, seq, K=16):
    """Mean first-order reward variance per site, sum_{s<=t} ||dL_t/dY_s||^2, over all t."""
    x, y = seq[:-1][None], seq[1:][None]
    T = x.shape[1]
    out = {s: torch.zeros(T, device=seq.device) for s in probe.sites}
    probe.active = True
    try:
        for c in range(0, T, K):
            tk = torch.arange(c, min(c + K, T), device=seq.device)
            k = len(tk)
            loss = per_token_losses(model, x.expand(k, -1).contiguous(), y.expand(k, -1).contiguous())
            loss.gather(1, tk[:, None]).sum().backward()
            before = torch.arange(T, device=seq.device)[None, :] <= tk[:, None]
            for s in probe.sites:
                g = probe.leaves[s].grad
                M = g.square().sum(-1)
                out[s][tk] = (M * before).sum(-1)
            probe.leaves.clear()
    finally:
        probe.active = False
    return {s: v.mean().item() for s, v in out.items()}


def sigma_for(model, site):
    if site.startswith('o@'):
        layer = int(site[2:])
        return 0.2 if layer < model.config.n_layer // 2 else 0.4
    return 0.2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/tinystories/validation.pt'))
    p.add_argument('--sequences', type=int, default=1)
    p.add_argument('--draws', type=int, default=256)
    p.add_argument('--steps', type=int, nargs='+', help='checkpoint steps to measure (default: all step*.pt)')
    p.add_argument('--sites', nargs='+', help='sites to measure (default: every probe site)')
    p.add_argument('--output', type=Path)
    args = p.parse_args()

    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')
    measure.AUTOCAST = False
    cfg = json.loads((args.run / 'config.json').read_text())
    config = GPTConfig(**cfg['model'])
    files = sorted(args.run.glob('step*.pt'), key=lambda q: int(q.stem[4:]))
    if args.steps:
        wanted = set(args.steps)
        files = [q for q in files if int(q.stem[4:]) in wanted]
    if not files:
        raise SystemExit(f'no checkpoints found in {args.run}')

    data = torch.load(args.data, weights_only=True)[:args.sequences].long().to(device)
    out_path = args.output or (args.run / 'eta.jsonl')
    with out_path.open('w') as log:
        for path in files:
            step = int(path.stem[4:])
            model = make_model(config, device, 42)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.requires_grad_(False)
            model.float()
            model.cos, model.sin = model.cos.float(), model.sin.float()
            probe = Probe(model)
            sites = args.sites or probe.sites
            sums = {s: [0.0, 0.0] for s in sites}  # measured, predicted, pooled over sequences
            for seq in data:
                pred = predicted_vars(model, probe, seq)
                for s in sites:
                    mean, var = empirical_check(model, seq, s, args.draws, sigma_for(model, s), 16)
                    sums[s][0] += var.mean().item()
                    sums[s][1] += pred[s]
            row = {}
            for s in sites:
                measured, predicted = sums[s][0], sums[s][1]
                if predicted <= 0:
                    continue
                row[s] = dict(eta=measured / predicted - 1, measured=measured, predicted=predicted,
                              sigma=sigma_for(model, s))
            log.write(json.dumps(dict(step=step, sites=row)) + '\n')
            log.flush()
            print(f'step {step}: ' + ' '.join(f'{s}={v["eta"]:+.3f}' for s, v in list(row.items())[:4]),
                  flush=True)
            del model, probe


if __name__ == '__main__':
    main()
