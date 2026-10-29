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
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig  # noqa: E402
from yoco.model_yoco import build  # noqa: E402


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
        for s in full:
            own[s][tk] = probe.leaves[s].grad[ar, tk]
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
                 norm=Gt.norm().item())
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
    per = [measure(model, probe, seq, args.K, arch == 'yoco', not args.fp32) for seq in data]
    keys = per[0].keys()
    summary = {s: {m: sum(q[s][m] for q in per) / len(per) for m in per[0][s]} for s in keys}
    res = dict(run=str(args.run), step=args.step, arch=arch, sequences=args.sequences, sites=summary)
    outp = args.run / f'bias_step{args.step}.json'
    outp.write_text(json.dumps(res, indent=1))
    for s, r in summary.items():
        print(f'{s:34s} ' + ' '.join(f'{m}={v:.4f}' for m, v in r.items() if m != 'norm'))


if __name__ == '__main__':
    main()
