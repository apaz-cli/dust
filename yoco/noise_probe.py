"""Per-site noise and bias of Dust's estimate, measured during training (a measuring stick: uses derivatives).

With `est.noise_probe = True`, DustX.finish keeps, for the last microbatch of a step, its inputs, Dust's per-token
error estimate at every site, and the weight gradients built from them (`est._probe`). `measure(model, est, part)`
then compares them with two references on the same microbatch:

- noise-free Dust: Dust's exact expectation (its own-token target, yoco/expected.py), per token at each site and
  per weight. Dust's estimate minus this is its noise.
- backprop: the true gradient, per weight. Noise-free Dust minus this is Dust's bias.

Per site (the output of one projection, or an MoE layer's routed pre-activations "moe" and chosen router logits
"gate"), logged under noise/<site>/...:
  rho      per-token noise: sum over tokens of |estimate - expectation|^2 / sum of |expectation|^2. The noise model
           predicts about C d (1 - R^2) / N ([[Noise simulator]] in the notes).
  ratio    |estimate| / |expectation| per token, summed the same way (1 when the scale is right)
Per weight (logged under wnoise/<parameter>/...):
  noise    |Dust - noise-free Dust|^2 / |noise-free Dust|^2 for that weight's gradient
  cos_bias cosine between noise-free Dust and backprop (1: no bias)
  cos      cosine between Dust and backprop (what training gets)
Costs two backward passes on one microbatch. The model's state (routing loads, requires_grad, grads) is restored.
"""
import torch
from torch.nn import functional as F

from yoco import expected


def _cos(a, b):
    return float(F.cosine_similarity(a.flatten().double(), b.flatten().double(), 0))


def _short(name):
    return name.replace('transformer.', '').replace('.weight', '')


def _site_grads(model, est, x, y, sites, use_expected):
    """Per-token gradients of the summed loss at each site's output, and the weight gradients, from one backward
    pass: the true ones, or noise-free Dust's (expected.install)."""
    mods = dict(model.named_modules())
    moes = {q: est.modules_moe[q] for q in getattr(est, 'moe_prefixes', [])}
    grabbed, hooks = {}, []
    for s in sites:
        if s in mods and not s.endswith('lm_head'):
            def hook(mod, inp, out, s=s):
                if torch.is_tensor(out) and out.requires_grad:
                    out.retain_grad()
                    grabbed[s] = out
            hooks.append(mods[s].register_forward_hook(hook))
    for m in moes.values():
        m.probe = {}
    loads = {q: m.load.clone() for q, m in moes.items()}
    if use_expected:
        expected.install(model, est.settings.gamma, cross_full=est.handoff)
    fast_head = model.lm_head.__dict__.pop('forward', None)  # fast16's fp32-output head mm has no autograd rule
    params = dict(model.named_parameters())
    try:
        with torch.enable_grad(), est.autocast():
            for p in params.values():
                p.requires_grad_(True)
                p.grad = None
            loss = type(model).forward(model, x, y, loss_reduction='sum')  # eager, not the compiled forward
            loss.backward()
        per_token = {s: t.grad.detach().float() for s, t in grabbed.items() if t.grad is not None}
        for q, m in moes.items():
            for kind, t in m.probe.items():
                if t.grad is not None:
                    g = t.grad.detach().float()
                    per_token[q + kind] = g.reshape(*est._probe['errors'][q + kind].shape) \
                        if q + kind in est._probe['errors'] else g
        weights = {n: p.grad.detach().float().clone() for n, p in params.items() if p.grad is not None}
    finally:
        for h in hooks:
            h.remove()
        for p in params.values():
            p.requires_grad_(False)
            p.grad = None
        for q, m in moes.items():
            m.probe = None
            m.load.copy_(loads[q])  # the probe's forward must not count toward load balancing
        if use_expected:
            for blk in model.transformer.h:
                blk.attn.__dict__.pop('forward', None)
            if hasattr(model, 'n_self'):
                model.__dict__.pop('cache', None)
        if fast_head is not None:
            model.lm_head.forward = fast_head
    return per_token, weights


def measure(model, est):
    """Noise and bias per site and per weight on the probed microbatch (see the module docstring)."""
    pr = est._probe
    x, y, errors, grads = pr['x'], pr['y'], pr['errors'], pr['grads']
    sites = list(errors)
    tok_exp, w_exp = _site_grads(model, est, x, y, sites, use_expected=True)
    _, w_bp = _site_grads(model, est, x, y, [], use_expected=False)
    out = {}
    for s, e in errors.items():
        ref = tok_exp.get(s)
        if ref is None or ref.shape != e.shape:
            continue
        den = ref.square().sum().clamp(min=1e-30)
        out[f'noise/{_short(s)}/rho'] = float((e - ref).square().sum() / den)
        out[f'noise/{_short(s)}/ratio'] = float(e.square().sum().sqrt() / den.sqrt())
    for n, g in grads.items():
        if n not in w_exp or n not in w_bp or g.shape != w_exp[n].shape:
            continue
        ge, gb = w_exp[n], w_bp[n]
        out[f'wnoise/{_short(n)}/noise'] = float((g - ge).square().sum() / ge.square().sum().clamp(min=1e-30))
        out[f'wnoise/{_short(n)}/cos_bias'] = _cos(ge, gb)
        out[f'wnoise/{_short(n)}/cos'] = _cos(g, gb)
    return {k: round(v, 5) for k, v in out.items()}
