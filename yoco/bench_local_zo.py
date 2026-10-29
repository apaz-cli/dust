"""Offline bench for derivative-free key/value (and YOCO cache) estimators.

DUST credits keys and values by jittering them, recomputing the reading attention, and scoring the change
in each query's output against that output's error e_u, credited back to the jittered row with decay
gamma. Every row is jittered at once, so the score at query u carries the noise of every row it reads:
cross-row interference, the local analogue of leakage.

Here e_u is the exact own-token o@ error (computed by autograd as a measurement), so the only noise left
is the local estimator's own. Its expectation is E_t = sum_u gamma^(u-t) (d o_u / d k_t)^T e_u, also
computed by autograd. Each design is scored by relative error of the weight-space update against E,
as a function of draws.

Designs (all derivative-free):
  dense      stock DUST: Gaussian noise on every row
  sparse-P   draw j jitters only rows t = j mod P (round robin), scaled by P; removes interference
             from rows closer than P
  anti       antithetic pairs (+a, -a), central difference
  rad        Rademacher noise
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, recipe  # noqa: E402
from model import attention, apply_rotary_emb, norm  # noqa: E402
from yoco import expected  # noqa: E402
from yoco.dustx import DustX  # noqa: E402
from yoco.model_yoco import build  # noqa: E402


def load(run):
    cfg = json.loads((Path(run) / 'config.json').read_text())
    m = build(cfg['args']['arch'], GPTConfig(**cfg['model']), torch.device('cuda'), 0)
    m.load_state_dict(torch.load(Path(run) / 'step3000.pt', map_location='cuda', weights_only=True))
    return m


def own_o_errors(model, x, y, gamma):
    """Exact own-token errors at every attention output (what DUST's o@ hubs estimate)."""
    expected.install(model, gamma)
    model.requires_grad_(True)
    grabbed, hooks = {}, []
    for i, blk in enumerate(model.transformer.h):
        def pre(mod, a, i=i):
            a[0].retain_grad()
            grabbed[i] = a[0]
        hooks.append(blk.attn.c_proj.register_forward_pre_hook(pre))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(x, y, loss_reduction='sum')
    loss.backward()
    for h in hooks:
        h.remove()
    for blk in model.transformer.h:
        del blk.attn.forward
    if hasattr(model, 'n_self'):
        del model.cache
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return {i: g.grad.float().detach() for i, g in grabbed.items()}


class Reader:
    """One jittered tensor (raw k or raw v of a self layer, or of the YOCO cache) and the attentions reading it."""

    def __init__(self, est, site, kind):
        self.est, self.kind = est, kind
        m, o = est.model, est.outputs
        cfg = m.config
        self.b, self.t = est.batch, est.length
        self.h, self.d = cfg.n_head, cfg.n_embd // cfg.n_head
        b, t, h, d = self.b, self.t, self.h, self.d
        cs = (m.cos[:, :t].float(), m.sin[:, :t].float())
        self.proc = lambda r: norm(apply_rotary_emb(r, *cs))
        if site == 'cache':
            self.raw_k = o['transformer.cache_k'].float().view(b, t, h, d)
            self.raw_v = o['transformer.cache_v'].float().view(b, t, h, d)
            self.layers = list(range(m.n_self, cfg.n_layer))
            self.vscale = None
        else:
            p = f'transformer.h.{site}.attn.'
            self.raw_k = o[p + 'c_k'].float().view(b, t, h, d)
            v = o[p + 'c_v'].float().view(b, t, h, d)
            self.vscale = None
            if str(site) in m.value_embeds:
                z = o[p + 've_gate'].float()
                ve = o[f'value_embeds.{site}'].float().view(b, t, h, d)
                v = v + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            self.raw_v = v
            self.layers = [site]
        self.q = {i: self.proc(o[f'transformer.h.{i}.attn.c_q'].float().view(b, t, h, d)) for i in self.layers}
        self.win = {i: m.window_sizes[i][0] for i in self.layers}
        self.sitename = {'cache': {'k': 'transformer.cache_k', 'v': 'transformer.cache_v'}}.get(
            site, {'k': f'transformer.h.{site}.attn.c_k', 'v': f'transformer.h.{site}.attn.c_v'})[kind]

    def outputs(self, k_raw, v):
        rep = k_raw.shape[0] // self.b
        kp = self.proc(k_raw)
        return {i: attention(self.q[i].repeat(rep, 1, 1, 1), kp, v, self.win[i]) for i in self.layers}

    def expected(self, e, gamma):
        """Autograd measurement: E_t = sum_u gamma^(u-t) (d o_u/d x_t)^T e_u for the jittered tensor x.
        Uses the verified weight route of yoco/expected.py with the probabilities held fixed."""
        t, sc = self.t, 1 / self.d ** 0.5
        pos = torch.arange(t, device='cuda')
        lag = pos[:, None] - pos[None, :]  # [u, s]
        x = (self.raw_k if self.kind == 'k' else self.raw_v).clone().requires_grad_()
        total = 0
        with torch.enable_grad():
            kw = (self.proc(x) if self.kind == 'k' else self.proc(self.raw_k)).transpose(1, 2)
            vw = (self.raw_v if self.kind == 'k' else x).transpose(1, 2)
            for i in self.layers:
                q = self.q[i].transpose(1, 2)
                mask = (lag >= 0) & (lag <= self.win[i])
                P = (q @ kw.detach().transpose(-1, -2) * sc).masked_fill(~mask, float('-inf')).softmax(-1)
                G = torch.where(mask, gamma ** lag.clamp(min=0).float(), torch.zeros((), device='cuda'))
                vd = vw.detach()
                od = P @ vd
                M = P * G
                MC = M * (q @ kw.transpose(-1, -2))
                Y = M @ vw + sc * (MC @ vd - MC.sum(-1, keepdim=True) * od)
                eu = e[i].view(self.b, t, self.h, self.d).transpose(1, 2)
                total = total + (Y * eu).sum()
            g, = torch.autograd.grad(total, x)
        return g.reshape(self.b, t, -1)

    def probs(self, i):
        """Forward attention probabilities of reading layer i, [b, h, u, s]."""
        t = self.t
        pos = torch.arange(t, device='cuda')
        lag = pos[:, None] - pos[None, :]
        mask = (lag >= 0) & (lag <= self.win[i])
        kp = self.proc(self.raw_k).transpose(1, 2)
        return (self.q[i].transpose(1, 2) @ kp.transpose(-1, -2) / self.d ** 0.5).masked_fill(~mask, float('-inf')).softmax(-1)

    def credit(self, i, mode, gamma):
        """Weight W[b, h, u, t] given to query u's score when crediting row t."""
        t = self.t
        pos = torch.arange(t, device='cuda')
        lag = pos[:, None] - pos[None, :]
        causal = ((lag >= 0) & (lag <= self.win[i])).float()
        if mode == 'gamma':
            return (causal * gamma ** lag.clamp(min=0).float()).expand(self.b, self.h, t, t)
        Pm = self.probs(i)
        if mode.startswith('attn'):  # attention-matched: w_ut = min(1, c * P_ut), c set so the mean weight on read rows is 1
            c = float(mode[4:] or 1) * t
            return (c * Pm).clamp(max=1.0)
        raise ValueError(mode)

    def estimate(self, e, gamma, draws, design, sigma=0.05, P=1, n=8, mode='gamma'):
        b, t, h, d = self.b, self.t, self.h, self.d
        pos = torch.arange(t, device='cuda')
        W = {i: self.credit(i, mode, gamma) for i in self.layers}
        base = self.outputs(self.raw_k, self.raw_v)
        rep = lambda z: z.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, d)
        est = torch.zeros(b, t, h, d, device='cuda')
        done = 0
        j = 0
        while done < draws:
            a = torch.randn(n, b, t, h, d, device='cuda')
            if design == 'rad':
                a = a.sign()
            if P > 1:
                r = (torch.arange(n, device='cuda') + j) % P
                a = a * (pos[None, :] % P == r[:, None]).float().view(n, 1, t, 1, 1)
            j += n
            signs = (1, -1) if design == 'anti' else (1,)
            delta = {}
            for sgn in signs:
                if self.kind == 'k':
                    out = self.outputs((rep(self.raw_k) + sgn * sigma * a.reshape(n * b, t, h, d)), rep(self.raw_v))
                else:
                    out = self.outputs(rep(self.raw_k), rep(self.raw_v) + sgn * sigma * a.reshape(n * b, t, h, d))
                for i in self.layers:
                    dd = out[i].view(n, b, t, h, d) - base[i]
                    delta[i] = delta.get(i, 0) + sgn * dd / len(signs)
            credited = 0
            for i in self.layers:
                sc_i = torch.einsum('buhd,nbuhd->nbuh', e[i].view(b, t, h, d), delta[i])
                credited = credited + torch.einsum('nbuh,bhut->nbth', sc_i, W[i])
            est += torch.einsum('nbth,nbthd->bthd', credited, a) / sigma * P
            done += n
        return est.reshape(b, t, -1) / draws


def cache_direct(est, rd, draws, sigma=0.05, n=8, attn_c=0.01, gamma=None):
    """YOCO only: jitter the raw cache rows, run the model, reward row t with sum_u W_ut (L_u - L'_u).
    W is attention-matched (mean over cross layers) or gamma^(u-t). One noise stage, no o@ errors."""
    b, t, h, d = rd.b, rd.t, rd.h, rd.d
    if gamma is None:
        W = sum((attn_c * t * rd.probs(i)).clamp(max=1.0) for i in rd.layers) / len(rd.layers)  # [b,h,u,t]
    else:
        pos = torch.arange(t, device='cuda')
        lag = pos[:, None] - pos[None, :]
        W = ((lag >= 0).float() * gamma ** lag.clamp(min=0).float()).expand(b, h, t, t)
    x, y = est.x.repeat(n, 1), est.y.repeat(n, 1)
    out = torch.zeros(b, t, h, d, device='cuda')
    for _ in range(draws // n):
        a = torch.randn(n, b, t, h * d, device='cuda')
        est.jitter = {rd.sitename: sigma * a.view(n * b, t, h * d)}
        try:
            L = est.losses(x, y).view(n, b, t)
        finally:
            est.jitter = {}
        r = (L - est.clean_loss.unsqueeze(0))
        r = r - r.mean(0, keepdim=True)
        cr = torch.einsum('nbu,bhut->nbth', r, W)
        out += torch.einsum('nbth,nbthd->bthd', cr, a.view(n, b, t, h, d)) / sigma
    return out.reshape(b, t, -1) / draws


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--gamma', type=float, default=0.98)
    p.add_argument('--draws', type=int, nargs='+', default=[16, 64, 256])
    p.add_argument('--designs', nargs='+', default=['dense', 'sparse4', 'sparse16', 'anti', 'rad'])
    p.add_argument('--sites', nargs='+', default=None)
    p.add_argument('--hub-draws', type=int, default=0, help='score against ZO-estimated o@ errors (real pipeline)')
    p.add_argument('--cache-direct', type=int, nargs='*', default=[], help='draw counts for the direct cache estimator')
    p.add_argument('--modes', nargs='+', default=['gamma0.98'], help='gammaG or attnC credit schemes')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    model = load(args.run)
    data = torch.load('data/tinystories/train.pt', weights_only=True).long()[:8].cuda()
    x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
    e = own_o_errors(model, x, y, args.gamma)
    est = DustX(model, recipe('10M', 256))
    est.capture(x, y)
    yoco = hasattr(model, 'n_self')
    e_ref = e
    if args.hub_draws:
        L = model.config.n_layer
        with torch.no_grad():
            e = {i: est.direct_errors([f'o@{i}'], args.hub_draws, 0.2 if i < L // 2 else 0.4)[f'o@{i}'] for i in range(L)}
        for i in range(L):
            c = F.cosine_similarity(e[i].flatten(), e_ref[i].flatten(), 0).item()
            print(f'o@{i} estimate vs exact cos {c:.3f}', flush=True)
    sites = args.sites or (['cache', 0, 1] if yoco else [0, 1, 2, 3])
    res = []
    for site in sites:
        site = site if site == 'cache' else int(site)
        for kind in ('k', 'v'):
            rd = Reader(est, site, kind)
            E = rd.expected(e_ref, 1.0)  # reference: the full local credit
            GE = est.reassemble(rd.sitename, E)
            for design in args.designs:
              P = int(design[6:]) if design.startswith('sparse') else 1
              for mode in args.modes:
                g = float(mode[5:]) if mode.startswith('gamma') else 1.0
                m = 'gamma' if mode.startswith('gamma') else mode
                for N in args.draws:
                    with torch.no_grad():
                        Z = rd.estimate(e, g, N, 'dense' if P > 1 else design, P=P, mode=m)
                    GZ = est.reassemble(rd.sitename, Z)
                    r = dict(site=str(site), kind=kind, design=design, mode=mode, draws=N,
                             rel=((GZ - GE).norm() / GE.norm()).item(),
                             cos=F.cosine_similarity(GZ.flatten(), GE.flatten(), 0).item())
                    res.append(r)
                    print(json.dumps(r), flush=True)
    for kind in ('k', 'v'):
        rd = Reader(est, 'cache', kind)
        GE = est.reassemble(rd.sitename, rd.expected(e_ref, 1.0))
        for N in args.cache_direct:
            for label, g in (('direct-attn', None), ('direct-g0.98', 0.98)):
                with torch.no_grad():
                    Z = cache_direct(est, rd, N, gamma=g)
                GZ = est.reassemble(rd.sitename, Z)
                r = dict(site='cache', kind=kind, design=label, mode='-', draws=N,
                         rel=((GZ - GE).norm() / GE.norm()).item(),
                         cos=F.cosine_similarity(GZ.flatten(), GE.flatten(), 0).item())
                res.append(r)
                print(json.dumps(r), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
