"""Exact (autograd) bias of what each DUST variant estimates, per site, on a checkpoint.

For a site with output Y and per-token losses L_t:
  true   g_t = dL/dY_t,  L = sum_u L_u                 (backprop)
  own    o_t = dL_t/dY_t                                (what stock DUST's per-token reward targets)
  hand   h_t = (dX_t/dY_t)^T gX_t,  gX = dL/dX exact    (YOCO handoff recipe, self-decoder sites only)
Weight-space targets are sum_t err_t x_t^T (scatter-add by token for embeddings). Reported:
  wcos_own / wcos_hand : cosine to the true weight gradient
  miss_own             : ||G_true - G_own|| / ||G_true||   (relative error of the expected update)
  fut                  : sum_t ||g_t - o_t||^2 / sum_t ||g_t||^2   (per-token future share)
Diagonal terms come from a replicated batch: copy k back-propagates only token t_k's objective.

With --gamma g, copy k back-propagates the discounted objective L~_t = sum_{u>=t} g^(u-t) L_u instead, which gives
a_ts = dL~_t/dY_s for every s. Reported per site: wcos_g (cos of sum_t a_tt x_t^T to the true weight gradient),
kappa_before = sum_{s<t} ||a_ts||^2 / sum ||a_tt||^2, kappa_after the same over s>t, and kappa_after_equal, the
article's equal-signal estimate sum_{s>t} g^(2(s-t)) (truncated at the sequence end). Exactly a_ts = g^(s-t) a_ss.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig  # noqa: E402
from yoco.model_yoco import build  # noqa: E402
from mlsweep.logger import MLSweepLogger  # noqa: E402  (no-op outside an mlsweep worker)


class Probe:
    def __init__(self, model):
        self.model, self.leaves, self.inputs, self.active = model, {}, {}, False
        self.sites = {}
        for name, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)) and name != 'lm_head':
                self.sites[name] = m
                m.register_forward_hook(self.hook(name))

    def hook(self, name):
        def h(m, args, out):
            if not self.active:
                return out
            z = torch.zeros(out.shape, device=out.device, dtype=torch.float32, requires_grad=True)
            self.leaves[name] = z
            self.inputs[name] = args[0].detach()
            return out + z.to(out.dtype)
        return h


def weight_grad(probe, name, err):
    x = probe.inputs[name][0]
    m = probe.sites[name]
    if isinstance(m, nn.Embedding):
        G = torch.zeros(m.weight.shape, device=err.device)
        return G.index_add_(0, x, err)
    return err.T @ x.float()


def measure_gamma(model, probe, seq, K, gamma, autocast):
    x, y = seq[:-1][None], seq[1:][None]
    T = x.shape[1]
    ctx = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast)
    probe.active = True
    with ctx():
        L = model(x, y, loss_reduction='none').view(-1).float()
    L.sum().backward()
    full = {s: probe.leaves[s].grad[0].clone() for s in probe.leaves}
    inputs = {s: probe.inputs[s] for s in probe.leaves}
    probe.leaves.clear()
    diag = {s: torch.zeros_like(full[s]) for s in full}
    acc = {s: [0.0, 0.0, 0.0] for s in full}   # signal, before, after
    pos = torch.arange(T, device=x.device)
    for c in range(0, T, K):
        tk = torch.arange(c, min(T, c + K), device=x.device)
        k = len(tk)
        ar = torch.arange(k, device=x.device)
        w = torch.where(pos[None] >= tk[:, None], gamma ** (pos[None] - tk[:, None]).clamp(min=0).float(), 0.0)
        with ctx():
            Lk = model(x.expand(k, -1).contiguous(), y.expand(k, -1).contiguous(), loss_reduction='none').float().view(k, -1)
        (Lk * w).sum().backward()
        for s in full:
            g = probe.leaves[s].grad
            diag[s][tk] = g[ar, tk].float()
            M = g.float().square().sum(-1)  # [k, T]
            acc[s][0] += M[ar, tk].sum().item()
            acc[s][1] += (M * (pos[None] < tk[:, None])).sum().item()
            acc[s][2] += (M * (pos[None] > tk[:, None])).sum().item()
        probe.leaves.clear()
    probe.active = False
    probe.inputs = inputs
    d = (pos[None] - pos[:, None]).float()
    equal = torch.where(d > 0, float(gamma) ** (2 * d), torch.zeros_like(d)).sum(1).mean().item()
    out = {}
    for s in full:
        Gt, Gg = weight_grad(probe, s, full[s]), weight_grad(probe, s, diag[s])
        sig, bef, aft = acc[s]
        out[s] = dict(wcos_g=F.cosine_similarity(Gt.flatten(), Gg.flatten(), 0).item(),
                      kappa_before=bef / max(sig, 1e-30), kappa_after=aft / max(sig, 1e-30), kappa_after_equal=equal)
    return out


def measure(model, probe, seq, K, yoco, autocast):
    x, y = seq[:-1][None], seq[1:][None]
    T = x.shape[1]
    ctx = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast)
    probe.active = True
    gX = None
    with ctx():
        if yoco:
            Xh = {}
            def hk(m, a, out):
                out = out * 1.0
                out.retain_grad()
                Xh['x'] = out
                return out
            hh = model.transformer.h[model.n_self - 1].register_forward_hook(hk)  # X before the cache: both paths
        L = model(x, y, loss_reduction='none').view(-1).float()
        if yoco:
            hh.remove()
    L.sum().backward()
    full = {s: probe.leaves[s].grad[0].clone() for s in probe.leaves}
    inputs = {s: probe.inputs[s] for s in probe.leaves}
    if yoco:
        gX = Xh['x'].grad[0].float().clone()
    probe.leaves.clear()

    own = {s: torch.zeros_like(full[s]) for s in full}
    leak = {s: 0.0 for s in full}
    ownsq = {s: 0.0 for s in full}
    hand = {s: torch.zeros_like(full[s]) for s in full} if yoco else None
    self_sites = [s for s in full if yoco and (s == 'transformer.wte' or s.startswith('value_embeds')
                  or (s.startswith('transformer.h.') and int(s.split('.')[2]) < model.n_self))]
    for c in range(0, T, K):
        tk = torch.arange(c, min(T, c + K), device=x.device)
        k = len(tk)
        ar = torch.arange(k, device=x.device)
        with ctx():
            Lk = model(x.expand(k, -1).contiguous(), y.expand(k, -1).contiguous(), loss_reduction='none').float().view(k, -1)
        Lk[ar, tk].sum().backward()
        before = torch.arange(T, device=x.device)[None, :] < tk[:, None]
        for s in full:
            g = probe.leaves[s].grad
            own[s][tk] = g[ar, tk]
            M = g.float().square().sum(-1)  # [k, T]: ||dL_tk / dY_s||^2
            leak[s] += (M * before).sum().item()
            ownsq[s] += M[ar, tk].sum().item()
        probe.leaves.clear()
        if yoco:
            with ctx():
                X = model(x.expand(k, -1).contiguous(), stop_at_handoff=True).float()
            (X[ar, tk] * gX[tk]).sum().backward()
            for s in self_sites:
                hand[s][tk] = probe.leaves[s].grad[ar, tk]
            probe.leaves.clear()
    probe.active = False
    probe.inputs = inputs
    out = {}
    for s in full:
        Gt = weight_grad(probe, s, full[s])
        Go = weight_grad(probe, s, own[s])
        r = dict(wcos_own=F.cosine_similarity(Gt.flatten(), Go.flatten(), 0).item(),
                 miss_own=((Gt - Go).norm() / Gt.norm()).item(),
                 fut=((full[s] - own[s]).square().sum() / full[s].square().sum()).item(),
                 norm=Gt.norm().item(), kappa=leak[s] / max(ownsq[s], 1e-30))
        if yoco and s in self_sites:
            Gh = weight_grad(probe, s, hand[s])
            r['wcos_hand'] = F.cosine_similarity(Gt.flatten(), Gh.flatten(), 0).item()
            r['miss_hand'] = ((Gt - Gh).norm() / Gt.norm()).item()
        out[s] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--step', type=int, default=3000)
    p.add_argument('--sequences', type=int, default=4)
    p.add_argument('--K', type=int, default=32)
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--gamma', type=float, help='measure the discounted-credit target and its leakage split')
    p.add_argument('--data', type=Path, default=Path('data/tinystories/validation.pt'))
    args = p.parse_args()
    cfg_json = json.loads((args.run / 'config.json').read_text())
    arch = cfg_json['args']['arch']
    config = GPTConfig(**cfg_json['model'])
    dev = torch.device('cuda')
    model = build(arch, config, dev, 0)
    model.load_state_dict(torch.load(args.run / f'step{args.step}.pt', map_location=dev, weights_only=True))
    model.requires_grad_(False)
    if args.fp32:
        model.transformer.wte.float()
        for ve in model.value_embeds.values():
            ve.float()
    probe = Probe(model)
    data = torch.load(args.data, weights_only=True).long()[:args.sequences].to(dev)
    if args.gamma is not None:
        per = [measure_gamma(model, probe, seq, args.K, args.gamma, not args.fp32) for seq in data]
    else:
        per = [measure(model, probe, seq, args.K, arch == 'yoco', not args.fp32) for seq in data]
    keys = per[0].keys()
    summary = {s: {m: sum(q[s][m] for q in per) / len(per) for m in per[0][s]} for s in keys}
    res = dict(run=str(args.run), step=args.step, arch=arch, gamma=args.gamma, sequences=args.sequences, sites=summary)
    out_dir = Path(os.environ['MLSWEEP_RUN_DIR']) if 'MLSWEEP_RUN_DIR' in os.environ else args.run
    outp = out_dir / (f'bias_step{args.step}.json' if args.gamma is None else f'bias_step{args.step}_g{args.gamma}.json')
    outp.write_text(json.dumps(res, indent=1))
    with MLSweepLogger() as logger:  # one record: '<metric>/<site>', site without the 'transformer.' prefix
        logger.log({f"{m}/{site.replace('transformer.', '')}": float(v) for site, r in summary.items()
                    for m, v in r.items() if m != 'norm'}, step=1)
    for s, r in summary.items():
        print(f'{s:34s} ' + ' '.join(f'{m}={v:.4f}' for m, v in r.items() if m != 'norm'))


if __name__ == '__main__':
    main()
