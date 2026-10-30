"""Measure what an optimal derivative-free draw allocation needs, per draw group, on a checkpoint.

For each group g with N_g draws in the recipe, two independent DUST estimates G1, G2 give
  noise   eps = (G1 - G2) / sqrt(2)                   one estimate's noise, Var = Sigma_g / N_g
  signal  |G_g|^2 ~= <G1, G2>                         unbiased when the two are independent
  B_g     = N_g ||G1 - G2||^2 / (2 <G1, G2>)          noise scale (draws for SNR 1)
  D_g     = N_g * ( [L(th + s eps) + L(th - s eps)] / 2 - L(th) )   per-draw noise damage, s = lr_g / (1 - beta) / tokens
  c_g     = seconds per draw (truncated, fast16)
The Neyman allocation that minimizes sum_g D_g / N_g at fixed sum_g c_g N_g is N_g* ∝ sqrt(D_g / c_g).
Everything is forward-only; no derivatives are used.

Groups: writers.l (attn.c_proj + mlp.c_proj, jittered together), hidden.l (mlp.c_fc), attn.l (q, k, v, gate,
ve and, for YOCO's cross layers, the cache, all driven by the o@ hub draws of block l), wte, head.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, make_optimizer, recipe  # noqa: E402
from yoco.dustx import DustX  # noqa: E402
from yoco.model_yoco import build  # noqa: E402


def settings_for(cfg_json):
    r = cfg_json['recipe']
    s = recipe('10M', 256)
    return replace(s, **{k: v for k, v in r.items() if k in s.__dataclass_fields__})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--step', type=int, default=3000)
    p.add_argument('--reps', type=int, default=3, help='independent estimate pairs per group')
    p.add_argument('--val', type=int, default=64, help='validation sequences for the damage probe')
    p.add_argument('--out')
    args = p.parse_args()
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('high')
    cj = json.loads((Path(args.run) / 'config.json').read_text())
    arch = cj['args']['arch']
    model = build(arch, GPTConfig(**cj['model']), torch.device('cuda'), 0)
    model.load_state_dict(torch.load(Path(args.run) / f'step{args.step}.pt', map_location='cuda', weights_only=True))
    s = settings_for(cj)
    opt = make_optimizer(model, s)
    lr_of = {}
    for g in opt.param_groups:
        for prm in g['params']:
            lr_of[id(prm)] = g['lr']
    beta = s.momentum
    est = DustX(model, s, seed=11)
    est.set_precision('fast16')
    L = model.config.n_layer
    train = torch.load('data/tinystories/train.pt', weights_only=True).long()
    batch = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(args.step))[:8]].cuda()
    x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
    val = torch.load('data/tinystories/validation.pt', weights_only=True).long()[:args.val].cuda()
    tokens = (y != -1).sum().item()
    params = dict(model.named_parameters())

    @torch.no_grad()
    def val_loss():
        tot = 0.0
        for b in val.split(32):
            tot += model(b[:, :-1].contiguous(), b[:, 1:].contiguous(), loss_reduction='sum').item()
        return tot / (val.shape[0] * (val.shape[1] - 1))

    def estimate(group):
        """One fresh DUST estimate of the group's weight gradients: {param name: tensor}, and seconds taken."""
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = {}
        if group == 'wte':
            e = est.direct_errors(['transformer.wte'], s.embedding, 0.2)
            out['transformer.wte.weight'] = est.reassemble('transformer.wte', e['transformer.wte'])
        elif group == 'head':
            out['lm_head.weight'] = est.reassemble('lm_head', est.head_error(s.head))
        else:
            kind, l = group.split('.')
            l = int(l)
            pre = f'transformer.h.{l}.'
            if kind == 'writers':
                e = est.direct_errors([pre + 'attn.c_proj', pre + 'mlp.c_proj'], s.writers[l], 0.2)
            elif kind == 'hidden':
                e = est.direct_errors([pre + 'mlp.c_fc'], s.hidden[l], 0.2)
            else:  # attention internals of block l, through its hub
                sigma = 0.2 if l < L // 2 else 0.4
                tg = {l: est.direct_errors([f'o@{l}'], s.hubs[l], sigma)[f'o@{l}']}
                e = {}
                if est.yoco and l >= est.n_self:
                    for k in ('q', 'k', 'v'):
                        r = est.zo_cross(k, s.local[k], {i: tg[l] if i == l else torch.zeros_like(tg[l])
                                                         for i in range(est.n_self, L)})
                        e.update({n: v for n, v in r.items() if n.endswith('c_q') and f'.{l}.' in n or 'cache' in n})
                else:
                    saved = est.model.config.n_layer
                    for k, c in s.local.items():
                        r = est.zo_self_local(k, c, {i: tg[l] if i == l else torch.zeros_like(tg[l])
                                                     for i in range(est.n_self if est.yoco else L)})
                        e.update({n: v for n, v in r.items() if f'.{l}.' in n or n == f'value_embeds.{l}'})
            for site, err in e.items():
                out[site + '.weight'] = est.reassemble(site, err)
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    groups = [f'{k}.{l}' for l in range(L) for k in ('writers', 'hidden', 'attn')] + ['wte', 'head']
    draws = {g: (s.embedding if g == 'wte' else s.head if g == 'head' else
                 {'writers': s.writers, 'hidden': s.hidden, 'attn': s.hubs}[g.split('.')[0]][int(g.split('.')[1])])
             for g in groups}
    with torch.no_grad():
        est.capture(x, y)
        L0 = val_loss()
        res = {}
        for g in groups:
            Bs, Ds, sig, ts, trs = [], [], [], [], []
            for _ in range(args.reps):
                G1, t1 = estimate(g)
                G2, t2 = estimate(g)
                ts += [t1, t2]
                diff2 = sum((G1[n] - G2[n]).square().sum().item() for n in G1)
                dot = sum((G1[n] * G2[n]).sum().item() for n in G1)
                sig.append(dot)
                lrg = sum(lr_of[id(params[n])] * (G1[n] - G2[n]).square().sum().item() for n in G1) / max(diff2, 1e-30)
                trs.append(lrg * draws[g] * diff2 / 2 / tokens ** 2)
                Bs.append(draws[g] * diff2 / (2 * dot) if dot > 0 else float('inf'))
                # damage of one estimate's noise, applied as a real update would be
                eps = {n: (G1[n] - G2[n]) / math.sqrt(2) for n in G1}
                scale = {n: lr_of[id(params[n])] / (1 - beta) / tokens for n in eps}
                orig = {n: params[n].detach().clone() for n in eps}
                vals = []
                for sgn in (1, -1):
                    for n in eps:
                        params[n].copy_(orig[n] + sgn * scale[n] * eps[n].to(params[n].dtype))
                    vals.append(val_loss())
                for n in eps:
                    params[n].copy_(orig[n])
                Ds.append(draws[g] * ((vals[0] + vals[1]) / 2 - L0))
            c = sum(ts) / len(ts) / draws[g]
            res[g] = dict(draws=draws[g], sec_per_draw=c, B=sorted(Bs)[len(Bs) // 2], signal=sum(sig) / len(sig),
                          D_per_draw=sum(Ds) / len(Ds), damage_now=sum(Ds) / len(Ds) / draws[g],
                          S_per_draw=sum(trs) / len(trs))
            print(f'{g:10s} N={draws[g]:5d} c={c * 1e3:6.3f}ms B={res[g]["B"]:10.1f} D/draw={res[g]["D_per_draw"]:+.5f}'
                  f' S/draw={res[g]["S_per_draw"]:.3e}', flush=True)
    # Neyman allocation at the same total time
    total = sum(r['draws'] * r['sec_per_draw'] for r in res.values())
    w = {g: math.sqrt(max(r['D_per_draw'], 0) / r['sec_per_draw']) for g, r in res.items()}
    norm_ = sum(w[g] * res[g]['sec_per_draw'] for g in res)
    for g in res:
        res[g]['neyman'] = total * w[g] / norm_ if norm_ > 0 else 0
        res[g]['neyman_ratio'] = res[g]['neyman'] / res[g]['draws']
    print('Neyman draws at equal time: ' + ' '.join(f'{g}={r["neyman"]:.0f}({r["neyman_ratio"]:.2f}x)' for g, r in res.items()))
    pred_now = sum(r['D_per_draw'] / r['draws'] for r in res.values())
    pred_ney = sum(r['D_per_draw'] / r['neyman'] for r in res.values() if r['neyman'] > 0)
    print(f'predicted total noise damage per step: current {pred_now:.5f}  neyman {pred_ney:.5f}')
    # stationary SGD model: excess loss ~ lr * tr(Sigma) / N, curvature-free
    wS = {g: math.sqrt(r['S_per_draw'] / r['sec_per_draw']) for g, r in res.items()}
    nS = sum(wS[g] * res[g]['sec_per_draw'] for g in res)
    for g in res:
        res[g]['neyman_stat'] = total * wS[g] / nS
    print('Stationary-model draws at equal time: ' + ' '.join(f'{g}={r["neyman_stat"]:.0f}({r["neyman_stat"] / r["draws"]:.2f}x)' for g, r in res.items()))
    s_now = sum(r['S_per_draw'] / r['draws'] for r in res.values())
    s_ney = sum(r['S_per_draw'] / r['neyman_stat'] for r in res.values())
    print(f'stationary excess-loss proxy: current {s_now:.4e}  allocated {s_ney:.4e}')
    out = dict(run=args.run, step=args.step, groups=res, total_sec=total, damage_now=pred_now, damage_neyman=pred_ney)
    Path(args.out or f'{args.run}/alloc_step{args.step}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
