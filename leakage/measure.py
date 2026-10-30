"""Measure DUST's cross-token leakage exactly, with autograd.

For a layer output Y (a DUST site) and per-token losses L_t, let M[t, s] = ||dL_t / dY_s||^2.
DUST rewards the jitter at token t with the loss change at token t, so to first order the
reward-weighted estimate at t has

    signal   = M[t, t]                      (own token)
    leakage  = sum_{s<t} M[t, s]            (earlier tokens' jitter, through attention)
    kappa_t  = leakage / signal

DUST also jitters attn.c_proj and mlp.c_proj of a block in the same pass, so their group
leakage adds the other site's sum_{s<=t} M_other[t, s].

Rows of M come from a replicated batch: copy k back-propagates only L_{t_k}, so one backward
gives K rows. Future-token bias is measured with one extra backward of sum_t L_t.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, make_model  # noqa: E402

LAG_EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


class Probe:
    """Adds a zero leaf to each site's output; its .grad is dL/dY."""

    def __init__(self, model):
        self.model, self.sites, self.leaves, self.inputs, self.hooks = model, [], {}, {}, []
        self.active = False
        mods = dict(model.named_modules())
        for i in range(model.config.n_layer):
            p = f'transformer.h.{i}.'
            for s in ('attn.c_proj', 'mlp.c_proj', 'mlp.c_fc'):
                self.sites.append(p + s)
                self.hooks.append(mods[p + s].register_forward_hook(self.out_hook(p + s)))
            self.sites.append(f'o@{i}')
            self.hooks.append(mods[p + 'attn.c_proj'].register_forward_pre_hook(self.in_hook(f'o@{i}')))
        self.sites.append('transformer.wte')
        self.hooks.append(mods['transformer.wte'].register_forward_hook(self.out_hook('transformer.wte')))

    def leaf(self, name, like, x):
        if not self.active:
            return like
        z = torch.zeros(like.shape, device=like.device, dtype=torch.float32, requires_grad=True)
        self.leaves[name] = z
        self.inputs[name] = x
        return like + z.to(like.dtype)

    def out_hook(self, name):
        def hook(m, args, out):
            return self.leaf(name, out, args[0].detach())
        return hook

    def in_hook(self, name):
        def hook(m, args):
            return (self.leaf(name, args[0], None),)
        return hook


AUTOCAST = True  # bf16, as DUST runs; pair_check.py --fp32 turns it off


def per_token_losses(model, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=AUTOCAST):
        return model(x, y, loss_reduction='none').view_as(y).float()


def lag_bins(T, device):
    lag = torch.arange(T, device=device)[:, None] - torch.arange(T, device=device)[None, :]
    b = torch.full_like(lag, -1)
    for j, lo in enumerate(LAG_EDGES[:-1]):
        b[(lag >= lo) & (lag < LAG_EDGES[j + 1])] = j
    return b  # [t, s] -> bin index, -1 for s >= t


def measure_sequence(model, probe, seq, rows, K, gen):
    device = seq.device
    x, y = seq[:-1][None], seq[1:][None]
    T = x.shape[1]
    stats = {}

    # Full-loss gradient for the future-token comparison.
    probe.active = True
    loss = per_token_losses(model, x, y)
    loss.sum().backward()
    full = {s: probe.leaves[s].grad[0].clone() for s in probe.sites}
    inputs = {s: (probe.inputs[s][0].float() if probe.inputs[s] is not None else None) for s in probe.sites}
    probe.leaves.clear()

    ts = torch.randperm(T, generator=gen, device='cpu')[:rows].sort().values.to(device)
    bins = lag_bins(T, device)
    acc = {s: dict(own=torch.zeros(len(ts), device=device), leak=torch.zeros(len(ts), device=device),
                   allsum=torch.zeros(len(ts), device=device),
                   lag=torch.zeros(len(LAG_EDGES) - 1, device=device),
                   own_vec=torch.zeros(len(ts), full[s].shape[-1], device=device)) for s in probe.sites}
    for c in range(0, len(ts), K):
        tk = ts[c:c + K]
        k = len(tk)
        loss = per_token_losses(model, x.expand(k, -1).contiguous(), y.expand(k, -1).contiguous())
        loss.gather(1, tk[:, None]).sum().backward()
        for s in probe.sites:
            g = probe.leaves[s].grad  # [k, T, d]
            M = g.square().sum(-1)  # [k, T]: M[t_k, s]
            ar = torch.arange(k, device=device)
            a = acc[s]
            a['own'][c:c + k] = M[ar, tk]
            before = torch.arange(T, device=device)[None, :] < tk[:, None]
            a['leak'][c:c + k] = (M * before).sum(-1)
            a['allsum'][c:c + k] = (M * (torch.arange(T, device=device)[None, :] <= tk[:, None])).sum(-1)
            b = bins[tk]  # [k, T]
            a['lag'] += torch.zeros(len(LAG_EDGES), device=device).index_add_(
                0, (b + 1).flatten(), M.flatten())[1:]
            a['own_vec'][c:c + k] = g[ar, tk]
        probe.leaves.clear()
    probe.active = False

    for s in probe.sites:
        a = acc[s]
        f = full[s][ts]  # true gradient at sampled tokens [R, d]
        ov = a['own_vec']
        st = dict(own=a['own'].sum().item(), leak=a['leak'].sum().item(),
                  kappa_t=(a['leak'] / a['own'].clamp_min(1e-30)).tolist(),
                  lag=a['lag'].tolist(), allsum=a['allsum'],
                  future=(f - ov).square().sum().item(), full=f.square().sum().item(),
                  tok_cos=torch.nn.functional.cosine_similarity(ov, f, dim=-1).mean().item())
        # Weight-gradient view: DUST's target sum_t g_tt x_t^T vs backprop's sum_t g_t x_t^T, over sampled t.
        xin = inputs[s] if inputs[s] is not None else None
        if s.startswith('o@'):
            xin = None
        if xin is not None and xin.is_floating_point():
            Gown, Gtrue = ov.T @ xin[ts], f.T @ xin[ts]
            st['weight_cos'] = torch.nn.functional.cosine_similarity(Gown.flatten(), Gtrue.flatten(), 0).item()
        stats[s] = st
    # Group leakage: attn.c_proj and mlp.c_proj are jittered together.
    for i in range(model.config.n_layer):
        A, B = f'transformer.h.{i}.attn.c_proj', f'transformer.h.{i}.mlp.c_proj'
        for me, other in ((A, B), (B, A)):
            stats[me]['group_leak'] = stats[me]['leak'] + stats[other]['allsum'].sum().item()
    for s in probe.sites:
        stats[s].pop('allsum')
        stats[s].setdefault('group_leak', stats[s]['leak'])
    return stats


@torch.no_grad()
def empirical_check(model, seq, site, draws, sigma, n=16):
    """Jitter one site as DUST does; return the per-token mean and variance of reward / sigma over draws."""
    x, y = seq[:-1][None], seq[1:][None]
    mods = dict(model.named_modules())
    clean = per_token_losses(model, x, y)
    if site.startswith('o@'):
        module, width = mods[f'transformer.h.{site[2:]}.attn.c_proj'], model.config.n_embd
    else:
        module = mods[site]
        width = module.weight.shape[0] if isinstance(module, nn.Linear) else module.weight.shape[1]
    z = {}
    if site.startswith('o@'):
        h = module.register_forward_pre_hook(lambda m, a: (a[0] + z['z'].to(a[0].dtype),))
    else:
        h = module.register_forward_hook(lambda m, a, o: o + z['z'].to(o.dtype))
    xs, ys = x.expand(n, -1).contiguous(), y.expand(n, -1).contiguous()
    rewards = []
    try:
        for _ in range(draws // n):
            z['z'] = sigma * torch.randn(n, x.shape[1], width, device=seq.device)
            rewards.append((clean - per_token_losses(model, xs, ys)) / sigma)
    finally:
        h.remove()
    r = torch.cat(rewards)  # [draws, T]
    return r.mean(0), r.var(0)


def summarize(per_seq, sites):
    """Pool per-sequence stats: ratios of sums over all measured tokens."""
    summary = {}
    for s in sites:
        own = sum(q[s]['own'] for q in per_seq)
        leak = sum(q[s]['leak'] for q in per_seq)
        gl = sum(q[s]['group_leak'] for q in per_seq)
        kappa_t = torch.tensor([r for q in per_seq for r in q[s]['kappa_t']])
        lag = torch.tensor([q[s]['lag'] for q in per_seq]).sum(0)
        summary[s] = dict(
            own=own, leak=leak, kappa=leak / own, kappa_group=gl / own,
            kappa_t_median=kappa_t.median().item(), kappa_t_p90=kappa_t.quantile(0.9).item(),
            lag_share=(lag / lag.sum().clamp_min(1e-30)).tolist(),
            future_over_own=sum(q[s]['future'] for q in per_seq) / own,
            tok_cos=sum(q[s]['tok_cos'] for q in per_seq) / len(per_seq),
            weight_cos=(sum(q[s]['weight_cos'] for q in per_seq) / len(per_seq)
                        if 'weight_cos' in per_seq[0][s] else None),
        )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True, help='a stepN.pt from train_small.py')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--sequences', type=int, default=8)
    p.add_argument('--rows', type=int, default=256, help='sampled target tokens t per sequence')
    p.add_argument('--K', type=int, default=16)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-draws', type=int, default=0, help='also run an empirical variance check')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')
    config = GPTConfig(**json.loads((args.checkpoint.parent / 'config.json').read_text())['model'])
    model = make_model(config, device, 42)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.requires_grad_(False)
    probe = Probe(model)
    data = torch.load(args.data, weights_only=True)[:args.sequences].long().to(device)
    gen = torch.Generator().manual_seed(args.seed)

    per_seq = []
    for j, seq in enumerate(data):
        per_seq.append(measure_sequence(model, probe, seq, args.rows, args.K, gen))
        print(f'seq {j} done', flush=True)

    summary = summarize(per_seq, probe.sites)
    out = dict(checkpoint=str(args.checkpoint), data=str(args.data), sequences=args.sequences,
               rows=args.rows, lag_edges=LAG_EDGES, sites=summary)

    if args.check_draws:
        checks = {}
        seq = data[0]
        last = config.n_layer - 1
        for site in ['transformer.h.0.mlp.c_fc', 'transformer.h.1.attn.c_proj', f'o@{last}',
                     f'transformer.h.{last}.mlp.c_proj']:
            sigma = 0.4 if site == f'o@{last}' else 0.2  # DUST's sigma for these sites
            mean, var = empirical_check(model, seq, site, args.check_draws, sigma, 16)
            # Predicted per-token variance of reward/sigma (first order): sum_{s<=t} M[t, s], all t.
            pr = predicted_var(model, probe, seq, site)
            ok = torch.isfinite(var)
            checks[site] = dict(sigma=sigma, measured_mean_var=var[ok].mean().item(),
                                predicted_mean_var=pr[ok].mean().item(),
                                corr=torch.corrcoef(torch.stack([var[ok], pr[ok]]))[0, 1].item(),
                                mean_reward=mean.mean().item())
            print(site, checks[site], flush=True)
        out['empirical'] = checks

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + '\n')
    for s, v in summary.items():
        print(f"{s:28s} kappa={v['kappa']:.3f} group={v['kappa_group']:.3f} med={v['kappa_t_median']:.3f} "
              f"fut/own={v['future_over_own']:.3f} wcos={v['weight_cos']}", flush=True)


def predicted_var(model, probe, seq, site, K=16):
    """sum_{s<=t} ||dL_t/dY_s||^2 for every t (first-order Var of reward/sigma)."""
    x, y = seq[:-1][None], seq[1:][None]
    T = x.shape[1]
    out = torch.zeros(T, device=seq.device)
    probe.active = True
    for c in range(0, T, K):
        tk = torch.arange(c, min(c + K, T), device=seq.device)
        k = len(tk)
        loss = per_token_losses(model, x.expand(k, -1).contiguous(), y.expand(k, -1).contiguous())
        loss.gather(1, tk[:, None]).sum().backward()
        g = probe.leaves[site].grad
        M = g.square().sum(-1)
        out[tk] = (M * (torch.arange(T, device=seq.device)[None, :] <= tk[:, None])).sum(-1)
        probe.leaves.clear()
    probe.active = False
    return out


if __name__ == '__main__':
    main()
