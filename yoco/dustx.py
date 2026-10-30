"""DUST with switchable parts, for the GPT and the YOCO model.

Options (all default to stock DUST behaviour where it applies):
  head='zo' | 'exact'      exact: dL/d(raw logits) = (softmax - onehot) * softcap', computed on cached logits
  local='zo' | 'vjp'       vjp: attention internals (q, k, v, gate, ve, and the YOCO cache) get the exact
                           vector-Jacobian product of their block's attention against the estimated o@ error.
                           A local derivative of one attention op on cached activations; nothing deeper.
  handoff=False | True     YOCO only. The self-decoder is not rewarded with token losses. Instead
                           g_X = (residual-path probe of the handoff, cache clean)  +  (exact pullback of the
                           cross-decoder o@ errors through the cache), and every self-decoder draw is scored
                           by -<g_X_t, dX_t> at its own token. Self-decoder draws run only the self-decoder.

In the YOCO cross-decoder, every site (writers, c_fc, o@, queries) is position-local given a clean cache,
so plain DUST there is unbiased with zero leakage.
"""
import torch
from torch import nn
from torch.nn import functional as F

from dust import DUST
from model import attention, apply_rotary_emb, has_ve, norm


class DustX(DUST):
    def __init__(self, model, settings, seed=42, head='zo', local='zo', handoff=False, handoff_draws=None,
                 oracle=(), credit='gamma', attn_c=0.01, res_gamma=0.0, res_blocks=2):
        super().__init__(model, settings, seed)
        self.res_gamma, self.res_blocks = res_gamma, res_blocks
        self.credit_mode, self.attn_c = credit, attn_c
        self.precision = 'bf16'
        self.truncate = True
        self._start = 0

        self.oracle = set(oracle)  # parameter groups that get their noise-free expected update instead
        self.head_mode, self.local_mode, self.handoff = head, local, handoff
        self.yoco = hasattr(model, 'n_self')
        self.n_self = model.n_self if self.yoco else model.config.n_layer
        self.handoff_draws = handoff_draws
        self.score = None  # None: token losses. Otherwise a per-token linear score of the handoff.
        if self.yoco:
            m = model.transformer.handoff
            self.hooks.append(m.register_forward_hook(self.output_hook('transformer.handoff')))

    def set_precision(self, mode):
        """'bf16' (stock), 'tf32' (fp32 activations, TF32 matmuls) or 'fast16' (fp16 matmuls and attention,
        fp32 residual stream, lm_head accumulated and returned in fp32)."""
        from contextlib import nullcontext
        self.precision = mode
        head = self.model.lm_head
        if mode == 'tf32':
            self.autocast = nullcontext
        elif mode == 'fast16':
            self.autocast = lambda: torch.autocast('cuda', dtype=torch.float16)
            def head_fwd(x, W=head.weight):
                with torch.autocast('cuda', enabled=False):
                    y = torch.mm(x.reshape(-1, x.shape[-1]).half(), W.half().t(), out_dtype=torch.float32)
                return y.view(*x.shape[:-1], -1)
            head.forward = head_fwd
        if mode in ('tf32', 'fast16'):
            self.model.transformer.wte.float()
            for ve in self.model.value_embeds.values():
                ve.float()

    # ---- rewards -------------------------------------------------------------------------------
    def capture(self, x, y):
        """Clean pass that also records the stream entering every block, for truncated draws."""
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.capturing = True
        self.state, self._reps, self._start = {}, {}, 0
        try:
            with self.autocast():
                self.clean_loss = self.model(x, y, loss_reduction='none', record=self.state).view_as(y).float()
        finally:
            self.capturing = False

    def start_for(self, sites):
        """First block a jitter at these sites can change; draws rerun the model from there."""
        if not self.truncate:
            return 0
        first = []
        for s in sites:
            if s.startswith('transformer.h.'):
                first.append(int(s.split('.')[2]))
            elif s.startswith('o@'):
                first.append(int(s[2:]))
            elif s == 'transformer.handoff':
                first.append(self.n_self)
            else:
                return 0
        return min(first)

    def rep_state(self, n):
        if n not in self._reps:
            st = self.state
            r = lambda t: t.repeat(n, *([1] * (t.dim() - 1)))
            self._reps[n] = {'x0': r(st['x0']), 'x': {i: r(t) for i, t in st['x'].items()}}
            if 'kv' in st:
                self._reps[n]['kv'] = tuple(r(t) for t in st['kv'])
        return self._reps[n]

    def losses(self, x, y):
        if self.score is None and self._start:
            with self.autocast():
                return self.model(x, y, loss_reduction='none', start=self._start,
                                  state=self.rep_state(x.shape[0] // self.batch)).view_as(y).float()
        if self.score is None:
            return super().losses(x, y)
        # Linearized loss through the handoff: <g_X, X> per token, X from the self-decoder alone.
        with self.autocast():
            X = self.model(x, stop_at_handoff=True).float()
        g = self.score.repeat(x.shape[0] // self.batch, 1, 1)
        return (X * g).sum(-1)

    def res_site(self, site):
        if site == 'transformer.wte':
            return True
        if site.startswith('o@'):
            return int(site[2:]) < self.res_blocks
        if site.startswith('transformer.h.'):
            return int(site.split('.')[2]) < self.res_blocks
        return False

    def direct_errors(self, sites, draws, sigma):  # DUST.direct_errors plus optional residual credit
        """One-sided loss drops, centered across each chunk of draws. """
        self._start = self.start_for(sites) if self.score is None else 0
        n = self.settings.nrep
        x, y = self.x.repeat(n, 1), self.y.repeat(n, 1)
        width = lambda s: self.model.config.n_embd if s.startswith('o@') else self.outputs[s].shape[-1]
        errors = {s: torch.zeros(self.batch, self.length, width(s), device=self.device) for s in sites}

        for _ in range(draws // n):
            noise = {s: self.noise(n * self.batch, self.length, width(s)) for s in sites}
            self.jitter = {s: sigma * a for s, a in noise.items()}
            try:
                perturbed_loss = self.losses(x, y).view(n, self.batch, self.length)
            finally:
                self.jitter = {}

            reward = self.clean_loss.unsqueeze(0) - perturbed_loss
            if n > 1:
                reward = reward - reward.mean(0, keepdim=True)
            if self.res_gamma and any(self.res_site(s) for s in sites):
                # credit token t's jitter with gamma-discounted later token losses (as DUST does for k/v)
                t = self.length
                pos = torch.arange(t, device=self.device)
                lag = pos[None, :] - pos[:, None]  # [t, u]
                K = (lag >= 0).float() * self.res_gamma ** lag.clamp(min=0).float()
                reward = torch.einsum('nbu,tu->nbt', reward, K)
            for site, a in noise.items():
                errors[site] -= torch.einsum(
                    'nbt,nbtd->btd', reward.to(self.dtype),
                    a.view(n, self.batch, self.length, -1),
                ).float() / sigma

        self._start = 0
        return {s: error / draws for s, error in errors.items()}

    # ---- head ----
    def head_error(self, draws, sigma=0.05):
        """DUST.head_error with the loss computed relative to the target logit (arXiv:2610.31642): per token,
        loss = log1p(sum over non-target j of exp(z_j - z_t)), so confident tokens keep their small rewards."""
        b, t = self.batch, self.length
        raw = self.outputs['lm_head']
        vocab, slab = self.model.config.vocab_size, 256
        slabs = vocab // slab
        # Head draws only touch cached logits, so batch many per round; centering is over the batch, per draw.
        n = max(d for d in (32, 16, 8, 4, 2, 1) if (draws // slabs) % d == 0)
        logits = 15 * torch.tanh(raw[..., :vocab].float() / 15)
        target = self.y.clamp(min=0)
        z_t = logits.gather(-1, target.unsqueeze(-1))                       # [b, t, 1]
        rel = torch.exp(logits - z_t)
        rel.scatter_(-1, target.unsqueeze(-1), 0.0)                         # non-target terms only
        slab_rel = rel.view(b, t, slabs, slab).sum(-1)                      # [b, t, slabs]
        total_rel = slab_rel.sum(-1)
        clean = torch.log1p(total_rel)
        tslab = target // slab
        error = torch.zeros_like(raw, dtype=torch.float32)
        rep = lambda x: x.repeat(n, *([1] * (x.dim() - 1)))
        for chunk in range(draws // n):
            blk = (chunk * n * slabs // draws) % slabs  # each slab gets draws // slabs draws, n at a time
            lo = blk * slab
            a = self.noise(n * b, t, slab)
            changed = 15 * torch.tanh((raw[..., lo:lo + slab].float().repeat(n, 1, 1) + sigma * a.float()) / 15)
            in_slab = rep(tslab == blk)                                     # target lies in this slab
            tgt = rep(target)
            zt_new = torch.where(in_slab, changed.gather(-1, (tgt - lo).clamp(0, slab - 1).unsqueeze(-1)).squeeze(-1),
                                 rep(z_t[..., 0]))
            others = rep(total_rel - slab_rel[..., blk]) * torch.exp(rep(z_t[..., 0]) - zt_new)
            mine = torch.exp(changed - zt_new.unsqueeze(-1))
            mine = torch.where(in_slab.unsqueeze(-1) & (torch.arange(lo, lo + slab, device=raw.device) == tgt.unsqueeze(-1)),
                               0.0, mine)
            loss = torch.log1p(others + mine.sum(-1)).view(n, b, t)
            reward = torch.where(self.y.unsqueeze(0) != -1, clean.unsqueeze(0) - loss, 0)
            if b > 1:
                reward = reward - reward.mean(1, keepdim=True)
            error[..., lo:lo + slab] -= torch.einsum(
                'nbt,nbtd->btd', (reward / sigma).to(self.dtype), a.view(n, b, t, slab),
            ).float()
        return error / (draws // slabs)

    def exact_head_error(self):
        raw = self.outputs['lm_head'].float()
        vocab = self.model.config.vocab_size
        capped = 15 * torch.tanh(raw[..., :vocab] / 15)
        p = capped.softmax(-1)
        target = self.y.clamp(min=0)
        p.scatter_add_(-1, target.unsqueeze(-1), -torch.ones_like(p[..., :1]))
        err = p * (1 - (capped / 15).square())
        err = err * (self.y != -1).unsqueeze(-1)
        out = torch.zeros_like(raw)
        out[..., :vocab] = err
        return out

    # ---- exact local VJPs ----------------------------------------------------------------------
    def _processed(self, raw, t):
        cos, sin = self.model.cos[:, :t].float(), self.model.sin[:, :t].float()
        return norm(apply_rotary_emb(raw, cos, sin))

    def vjp_self_layer(self, i, target):
        """Exact local gradient of block i's attention w.r.t. its q, k, v, gate and value embedding."""
        cfg = self.model.config
        b, t, h, d = self.batch, self.length, cfg.n_head, cfg.n_embd // cfg.n_head
        prefix = f'transformer.h.{i}.attn.'
        leaves = {n: self.outputs[prefix + 'c_' + n].float().view(b, t, h, d).requires_grad_() for n in 'qkv'}
        ve_site = str(i) in self.model.value_embeds
        if ve_site:
            leaves['gate'] = self.outputs[prefix + 've_gate'].float().requires_grad_()
            leaves['ve'] = self.outputs[f'value_embeds.{i}'].float().view(b, t, h, d).requires_grad_()
        with torch.enable_grad():
            values = leaves['v']
            if ve_site:
                values = values + (2 * torch.sigmoid(leaves['gate'])).unsqueeze(-1) * leaves['ve']
            out = attention(self._processed(leaves['q'], t), self._processed(leaves['k'], t), values,
                            self.model.window_sizes[i][0])
            grads = torch.autograd.grad(out, list(leaves.values()), grad_outputs=target.view(b, t, h, d))
        names = {'q': prefix + 'c_q', 'k': prefix + 'c_k', 'v': prefix + 'c_v',
                 'gate': prefix + 've_gate', 've': f'value_embeds.{i}'}
        return {names[n]: g.reshape(b, t, -1) for n, g in zip(leaves, grads)}

    def vjp_cross(self, targets, want_handoff):
        """Exact local gradient of every cross-decoder attention w.r.t. its queries and the shared cache.

        With want_handoff the graph starts at the handoff X, so it also returns the cache path's dL/dX.
        """
        cfg, m = self.model.config, self.model
        b, t, h, d = self.batch, self.length, cfg.n_head, cfg.n_embd // cfg.n_head
        cross = range(self.n_self, cfg.n_layer)
        qs = {i: self.outputs[f'transformer.h.{i}.attn.c_q'].float().view(b, t, h, d).requires_grad_()
              for i in cross}
        with torch.enable_grad():
            if want_handoff:
                X = self.inputs['transformer.handoff'].float().requires_grad_()
                xn = norm(X)
                kr = F.linear(xn, m.transformer.cache_k.weight.float()).view(b, t, h, d)
                vr = F.linear(xn, m.transformer.cache_v.weight.float()).view(b, t, h, d)
            else:
                kr = self.outputs['transformer.cache_k'].float().view(b, t, h, d)
                vr = self.outputs['transformer.cache_v'].float().view(b, t, h, d)
            kr.requires_grad_() if not want_handoff else kr.retain_grad()
            vr.requires_grad_() if not want_handoff else vr.retain_grad()
            kp = self._processed(kr, t)
            total = 0
            for i in cross:
                out = attention(self._processed(qs[i], t), kp, vr, m.window_sizes[i][0])
                total = total + (out * targets[i].view(b, t, h, d)).sum()
            leaves = [kr, vr] + list(qs.values()) + ([X] if want_handoff else [])
            grads = torch.autograd.grad(total, leaves)
        res = {'transformer.cache_k': grads[0].reshape(b, t, -1), 'transformer.cache_v': grads[1].reshape(b, t, -1)}
        for i, g in zip(cross, grads[2:2 + len(qs)]):
            res[f'transformer.h.{i}.attn.c_q'] = g.reshape(b, t, -1)
        dX = grads[-1] if want_handoff else None
        return res, dX

    # ---- derivative-free local estimators ------------------------------------------------------
    def attn_credit(self, qp, kp, window):
        """Attention-matched credit W[b, h, u, t] = min(1, c * T * P_ut): query u's score goes to row t in
        proportion to how much u reads t. Uses only forward attention probabilities."""
        b, t, h, d = qp.shape
        pos = torch.arange(t, device=qp.device)
        lag = pos[:, None] - pos[None, :]
        mask = (lag >= 0) & (lag <= window)
        P = (qp.float().transpose(1, 2) @ kp.float().transpose(1, 2).transpose(-1, -2) / d ** 0.5)
        P = P.masked_fill(~mask, float('-inf')).softmax(-1)
        return (self.attn_c * t * P).clamp(max=1.0)

    def local_errors(self, kind, draws, targets, sigma=0.05):  # DUST.local_errors with a pluggable credit kernel
        """Score attention-output changes against estimated output errors (no derivatives).

        Each draw recomputes the block's attention output from the cached clean activations and scores the
        change per head against the output's estimated error (`targets`, computed by direct_errors on the 
        o@{layer} sites). Queries keep their own token's score; the other kinds sum the scores of later
        tokens, decayed by gamma per token.
        """
        cfg = self.model.config
        n, b, t, h, d = self.settings.nrep, self.batch, self.length, cfg.n_head, cfg.n_embd // cfg.n_head
        site_suffix = {'q': 'attn.c_q', 'k': 'attn.c_k', 'v': 'attn.c_v', 'gate': 'attn.ve_gate'}
        layers = [i for i in range(cfg.n_layer) if kind not in ('ve', 'gate') or has_ve(i, cfg.n_layer)]
        sites = {
            i: f'value_embeds.{i}' if kind == 've' else f'transformer.h.{i}.{site_suffix[kind]}'
            for i in layers
        }
        width = h if kind == 'gate' else cfg.n_embd
        errors = {i: torch.zeros(b, t, width, device=self.device) for i in layers}
        cos, sin = self.model.cos[:, :t].float(), self.model.sin[:, :t].float()

        def processed(raw):
            return norm(apply_rotary_emb(raw, cos, sin)).to(self.dtype)

        def repeat(x):
            return x.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, d)

        clean = {}
        for i in layers:
            prefix = f'transformer.h.{i}.attn.'
            q, k, v = [self.outputs[prefix + 'c_' + name].float().view(b, t, h, d) for name in ('q', 'k', 'v')]
            z = self.outputs[prefix + 've_gate'].float() if has_ve(i, cfg.n_layer) else None
            ve = self.outputs[f'value_embeds.{i}'].float().view(b, t, h, d) if z is not None else None
            values = v if z is None else v + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            qp, kp = processed(q), processed(k)
            window = self.model.window_sizes[i][0]
            baseline = attention(qp, kp, values.to(self.dtype), window)
            positions = torch.arange(t, device=self.device)
            lag = positions[None, :] - positions[:, None]  # [perturbation position, output position]
            # credit[t, s]: how much of output s's score goes to the jitter at t, inside the causal window
            credit = ((lag >= 0) & (lag <= window)).float() * self.settings.gamma ** lag.clamp(min=0)
            if self.credit_mode == 'attn':
                credit = self.attn_credit(qp, kp, window)
            clean[i] = q, k, z, ve, qp, kp, values, baseline, credit

        for _ in range(draws // n):
            for i in layers:
                a = self.noise(n, b, t, width, dtype=torch.float32)
                q, k, z, ve, qp, kp, values, baseline, credit = clean[i]
                window = self.model.window_sizes[i][0]
                divisor = sigma
                if kind in ('q', 'k'):
                    raw = q if kind == 'q' else k
                    perturbed = processed(
                        (raw.unsqueeze(0) + sigma * a.view(n, b, t, h, d)).reshape(n * b, t, h, d)
                    )
                    qa, ka = (perturbed, repeat(kp)) if kind == 'q' else (repeat(qp), perturbed)
                    delta = (
                        attention(qa, ka, repeat(values.to(self.dtype)), window).view(n, b, t, h, d)
                        - baseline
                    )
                else:
                    if kind == 'gate':
                        delta_gate = 2 * torch.sigmoid(z + sigma * a) - 2 * torch.sigmoid(z - sigma * a)
                        delta_value = delta_gate.unsqueeze(-1) * ve
                        divisor = 2 * sigma
                    else:
                        delta_value = sigma * a.view(n, b, t, h, d)
                        if kind == 've':
                            delta_value = delta_value * (2 * torch.sigmoid(z)).unsqueeze(-1)
                    delta = attention(
                        repeat(qp), repeat(kp),
                        delta_value.reshape(n * b, t, h, d).to(self.dtype), window,
                    ).view(n, b, t, h, d)

                score = torch.einsum('bthd,nbthd->nbth', targets[i].view(b, t, h, d).to(self.dtype), delta)
                if kind != 'q' and self.credit_mode == 'attn':
                    score = torch.einsum('nbuh,bhut->nbth', score.float(), credit)
                elif kind != 'q':
                    score = torch.einsum('nbsh,ts->nbth', score.float(), credit)
                estimate = torch.einsum('nbth,nbthd->bthd', score.float(), a.view(n, b, t, h, -1))
                errors[i] += estimate.reshape(b, t, width) / divisor

        return {sites[i]: errors[i] / draws for i in layers}

    # ---- ZO local estimators for the YOCO cross-decoder (stock-DUST analogue) -------------------
    def zo_cross(self, kind, draws, targets, sigma=0.05):
        """kind 'q': each cross layer's queries, own-token score. 'k'/'v': the shared cache, scored against
        every cross layer's o@ error and credited to the jittered row with decay gamma, as DUST does for k/v."""
        cfg, m = self.model.config, self.model
        n, b, t, h, d = self.settings.nrep, self.batch, self.length, cfg.n_head, cfg.n_embd // cfg.n_head
        cross = list(range(self.n_self, cfg.n_layer))
        rep = lambda x: x.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, d)
        kr = self.outputs['transformer.cache_k'].float().view(b, t, h, d)
        vr = self.outputs['transformer.cache_v'].float().view(b, t, h, d).to(self.dtype)
        kp = self._processed(kr, t).to(self.dtype)
        qp = {i: self._processed(self.outputs[f'transformer.h.{i}.attn.c_q'].float().view(b, t, h, d), t)
              .to(self.dtype) for i in cross}
        base = {i: attention(qp[i], kp, vr, m.window_sizes[i][0]) for i in cross}
        pos = torch.arange(t, device=self.device)
        lag = pos[None, :] - pos[:, None]
        credit = (lag >= 0).float() * self.settings.gamma ** lag.clamp(min=0)
        W = {i: self.attn_credit(qp[i], kp, t) for i in cross} if self.credit_mode == 'attn' and kind != 'q' else None
        if kind == 'q':
            errors = {i: torch.zeros(b, t, cfg.n_embd, device=self.device) for i in cross}
        else:
            err = torch.zeros(b, t, cfg.n_embd, device=self.device)
        for _ in range(draws // n):
            if kind == 'q':
                for i in cross:
                    a = self.noise(n, b, t, h * d, dtype=torch.float32)
                    raw = self.outputs[f'transformer.h.{i}.attn.c_q'].float().view(b, t, h, d)
                    qa = self._processed((raw.unsqueeze(0) + sigma * a.view(n, b, t, h, d)).reshape(n * b, t, h, d), t)
                    delta = attention(qa.to(self.dtype), rep(kp), rep(vr), t).view(n, b, t, h, d) - base[i]
                    score = torch.einsum('bthd,nbthd->nbth', targets[i].view(b, t, h, d).to(self.dtype), delta)
                    errors[i] += torch.einsum('nbth,nbthd->bthd', score.float(),
                                              a.view(n, b, t, h, d)).reshape(b, t, -1) / sigma
                continue
            a = self.noise(n, b, t, h * d, dtype=torch.float32)
            score = 0
            credited = 0
            for i in cross:
                if kind == 'k':
                    ka = self._processed((kr.unsqueeze(0) + sigma * a.view(n, b, t, h, d)).reshape(n * b, t, h, d), t)
                    delta = attention(rep(qp[i]), ka.to(self.dtype), rep(vr), t).view(n, b, t, h, d) - base[i]
                else:
                    delta = attention(rep(qp[i]), rep(kp), (sigma * a).view(n * b, t, h, d).to(self.dtype),
                                      t).view(n, b, t, h, d)
                s_i = torch.einsum('bthd,nbthd->nbth', targets[i].view(b, t, h, d).to(self.dtype), delta).float()
                if self.credit_mode == 'attn':
                    credited = credited + torch.einsum('nbuh,bhut->nbth', s_i, W[i])
                else:
                    score = score + s_i
            score = credited if self.credit_mode == 'attn' else torch.einsum('nbsh,ts->nbth', score, credit)
            err += torch.einsum('nbth,nbthd->bthd', score, a.view(n, b, t, h, d)).reshape(b, t, -1) / sigma
        if kind == 'q':
            return {f'transformer.h.{i}.attn.c_q': e / draws for i, e in errors.items()}
        return {f'transformer.cache_{kind}': err / draws}

    def zo_self_local(self, kind, draws, targets):
        """Stock DUST local_errors restricted to the self-decoder (YOCO) by hiding the cross layers."""
        if not self.yoco:
            return self.local_errors(kind, draws, targets)
        cfg = self.model.config
        saved = cfg.n_layer
        cfg.n_layer = self.n_self
        try:
            return self.local_errors(kind, draws, targets)
        finally:
            cfg.n_layer = saved

    # ---- noise-free expected update for selected groups --------------------------------------
    @staticmethod
    def group(name):
        if name == 'lm_head.weight':
            return 'head'
        if name == 'transformer.wte.weight':
            return 'wte'
        if 'c_proj' in name:
            return 'writers'
        if 'c_fc' in name:
            return 'hidden'
        if 'lambdas' in name:
            return 'scalars'
        return 'attn'

    def expected_grads(self, x, y):
        from yoco import expected
        m = self.model
        expected.install(m, self.settings.gamma, cross_full=self.handoff)
        params = {n: p for n, p in m.named_parameters() if self.group(n) in self.oracle}
        try:
            with torch.enable_grad(), self.autocast():
                for p in params.values():
                    p.requires_grad_(True)
                loss = m(x, y, loss_reduction='sum')
                gs = torch.autograd.grad(loss, list(params.values()))
        finally:
            for p in params.values():
                p.requires_grad_(False)
            for blk in m.transformer.h:
                del blk.attn.forward
            if self.yoco:
                del m.cache
        return {n: g.float() for n, g in zip(params, gs)}

    # ---- the update ----------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, x, y, optimizer):
        self.capture(x, y)
        r, cfg = self.settings, self.model.config
        L, S = cfg.n_layer, self.n_self
        grads = {}
        errors = {}

        def direct(sites, count, sigma):
            if count > 0:
                errors.update(self.direct_errors(sites, count, sigma))

        def hub(layer):
            sigma = 0.2 if layer < L // 2 else 0.4
            return self.direct_errors([f'o@{layer}'], r.hubs[layer], sigma)[f'o@{layer}']

        # Cross-decoder (YOCO) or the whole model (GPT): token-loss rewards.
        first = S if (self.yoco and self.handoff) else 0
        for layer in range(first, L):
            p = f'transformer.h.{layer}.'
            direct([p + 'attn.c_proj', p + 'mlp.c_proj'], r.writers[layer], 0.2)
            direct([p + 'mlp.c_fc'], r.hidden[layer], 0.2)
        targets = {layer: hub(layer) for layer in range(first, L)}

        if self.yoco and self.handoff:
            # g_X: residual path (cache held clean) by probe, plus the cache path by exact pullback.
            nh = self.handoff_draws or 2 * r.writers[S]
            g_res = self.direct_errors(['transformer.handoff'], nh, 0.2)['transformer.handoff']
            cache_err, dX = self.vjp_cross(targets, want_handoff=True)
            errors.update(cache_err)
            self.score = g_res + dX
            clean_loss = self.clean_loss
            self.clean_loss = (self.inputs['transformer.handoff'].float() * self.score).sum(-1)
            try:
                for layer in range(S):
                    p = f'transformer.h.{layer}.'
                    direct([p + 'attn.c_proj', p + 'mlp.c_proj'], r.writers[layer], 0.2)
                    direct([p + 'mlp.c_fc'], r.hidden[layer], 0.2)
                for layer in range(S):
                    targets[layer] = hub(layer)
                direct(['transformer.wte'], r.embedding, 0.2)
            finally:
                self.score = None
                self.clean_loss = clean_loss
        else:
            direct(['transformer.wte'], r.embedding, 0.2)
            if self.yoco:
                if self.local_mode == 'vjp':
                    errors.update(self.vjp_cross(targets, want_handoff=False)[0])
                else:
                    errors.update(self.zo_cross('q', r.local['q'], targets))
                    errors.update(self.zo_cross('k', r.local['k'], targets))
                    errors.update(self.zo_cross('v', r.local['v'], targets))

        # Self-attention internals (all GPT blocks, or the YOCO self-decoder).
        if self.local_mode == 'vjp':
            for layer in range(S):
                errors.update(self.vjp_self_layer(layer, targets[layer]))
        else:
            for kind, count in r.local.items():
                errors.update(self.zo_self_local(kind, count, targets))

        if self.head_mode == 'exact':
            errors['lm_head'] = self.exact_head_error()
        else:
            errors['lm_head'] = self.head_error(r.head)

        for site, error in errors.items():
            grads[site + '.weight'] = self.reassemble(site, error)
        grads.update(self.scalar_gradients(8))
        if self.oracle:
            grads.update(self.expected_grads(x, y))

        valid = (y != -1).sum()
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.model.named_parameters():
            g = (grads[name] / valid).to(parameter.dtype)
            if not torch.isfinite(g).all():
                raise FloatingPointError(f'Non-finite estimate for {name}')
            parameter.grad = g
        optimizer.step()
        return self.clean_loss[y != -1].mean().item()
