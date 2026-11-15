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
import re

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
        self.cache_gamma = None  # credit decay for the shared cache's k/v (default: settings.gamma)
        self.credit_mode, self.attn_c = credit, attn_c
        self.precision = 'bf16'
        self.truncate = True
        self._start = 0
        self.linear_v = False        # value errors from the linear attention map (forward probabilities), no draws
        self.kv_credit = False
        self.pool = False
        self.chain = False           # local chaining (implies pooling): no hidden or hub full-forward draws
        self.chain_draws = 256
        self.chain_pool = True
        self.chain_hidden = True     # False: only the hub (o@) errors are chained; MLP hidden keeps direct draws       # score local probes against pooled (biased) or own (unbiased) stream errors            # residual pooling of writer estimates across depths
        self.pool_decay = 0.9
        self._halves, self._pool_ema = {}, {}
        self.delta_fmt = None        # 'fp8' / 'nvfp4': emulate low-precision delta forwards in every draw
        self._qw = {}
        self.kv_credit_values_only = False
        self.kv_credit_embedding = False       # augment upstream rewards with cross-token credit through estimated K/V errors
        self.kv_strict = {}          # module name -> error at its output, credited from strictly later queries
        self._aug, self._draw_outs = {}, {}
        self.live_fc = False         # MLP hidden draws jitter only units with positive clean pre-activation
        self.cv = False              # control variate: per-site ridge memory keyed on the finite-difference head error
        self.cv_beta, self.cv_warmup = 0.9, 20
        self._cv, self._cv_P, self._cv_key, self._cv_steps = {}, None, None, 0
        self.cv_stats = {}           # site -> [sum raw reward^2, sum corrected reward^2] since last read

        self.oracle = set(oracle)  # parameter groups that get their noise-free expected update instead
        self.oracle_re = None  # or a regex on parameter names (diagnostic: which sites' noise costs loss)
        from yoco.moe import MoE
        self.modules_moe = {n + '.': m for n, m in model.named_modules() if isinstance(m, MoE)}
        self.moe_prefixes = list(self.modules_moe)
        for q, m in self.modules_moe.items():
            assert m.hs, 'Dust MoE needs a shared expert: its c_proj output is the layer-output site'
            m.dust = (self, q)
        self.accum, self.accum_valid = None, 0  # gradient accumulation over microbatches (train.py --microbatch-size)
        self.readout = False  # cross-token readout: reward each draw's token t also for later tokens' predicted loss change
        self.readout_targets = None  # {layer: attention-output (o@) errors [b, t, n_embd]} the readout scores against
        self.readout_scale = 1.0  # weight of the readout term in each draw's reward
        # 'credit': add to token t the predicted change of later tokens' losses from its own keys and values;
        # 'leak': subtract from token s the predicted change of its own loss from earlier tokens' keys and values
        # (the leakage into its reward; unbiased whatever the targets, since it depends only on other tokens'
        # noise); 'both': the two together ("symmetric readout")
        self.readout_mode = 'credit'
        # anchored draws: sites below self-decoder block `anchor` rerun only up to it and score the change of the
        # stream entering it against the error estimated there (block anchor-1's mlp.c_proj error)
        self.anchor, self.anchor_scale, self._anchor_err = 0, 1, None
        self.readout_warmup, self._steps = 0, 0  # steps before the readout turns on (its targets start as noise)
        self._ro_names, self._ro_clean = set(), {}
        self.freeze_attn = False  # residual-stream draws: hold every later attention output (and the cache) clean
        self._frozen = set()
        self.cache_separate = False  # the same leak-free scoring for the YOCO cache k/v (all cross layers)
        self.local_separate = False  # leak-free local k/v/gate/ve: each key's jitter scored alone (closed form)
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
        self._ro_clean = {}
        self.state, self._reps, self._start = {}, {}, 0
        try:
            with self.autocast():
                self.clean_loss = self.model(x, y, loss_reduction='none', record=self.state).view_as(y).float()
        finally:
            self.capturing = False
        if self.cv:
            k = self.cv_key()
            kf = k.reshape(-1, k.shape[-1])
            P = kf.T @ kf / kf.shape[0]
            self._cv_P = P if self._cv_P is None else self.cv_beta * self._cv_P + (1 - self.cv_beta) * P
            self._cv_key = k
            self._cv_steps += 1

    # ---- control variate (associative credit memory, noise PR #6) -------------------------------
    def cv_key(self, h=1e-2):
        """Head error by central finite differences of each token's loss along every raw logit of the cached
        logits (no draws), mapped to the stream width by the head's weights. The key only shapes the
        control variate; the estimate's expectation does not depend on it."""
        V = self.model.config.vocab_size
        raw = self.outputs['lm_head'][..., :V].float()
        cap = lambda z: 15 * torch.tanh(z / 15)
        c = cap(raw)
        m = c.max(-1, keepdim=True).values
        e = torch.exp(c - m)
        Z = e.sum(-1, keepdim=True)
        tgt = self.y.clamp(min=0).unsqueeze(-1)

        def change(sign):
            moved = cap(raw + sign * h)
            d = torch.log1p((torch.exp(moved - m) - e) / Z)
            return d.scatter_add_(-1, tgt, -(moved.gather(-1, tgt) - c.gather(-1, tgt)))

        err = (change(1) - change(-1)) / (2 * h) * (self.y != -1).unsqueeze(-1)
        return err @ self.model.lm_head.weight[:V].float()

    def cv_predict(self, site):
        """Ridge retrieval M k for every token, from earlier steps' estimates; None before any are stored."""
        st = self._cv.get(site)
        if st is None:
            return None
        P = self._cv_P
        if not (torch.isfinite(P).all() and torch.isfinite(st['S']).all()):
            return None  # a non-finite memory (after a bad step): no prediction this step, the draws do it all
        reg = (0.1 * P.diagonal().mean() + 1e-12) * torch.eye(P.shape[0], device=P.device)
        try:
            M = torch.linalg.solve(P + reg, st['S'].T).T
        except torch.linalg.LinAlgError:
            return None  # singular even with the ridge term: skip the prediction rather than crash
        st['raw'] = self._cv_key @ M.T
        return st['raw']

    def cv_update(self, site, g):
        """Store this step's (unbiased) estimate in the memory, and refit the trust scale out of sample."""
        k = self._cv_key.reshape(-1, self._cv_key.shape[-1])
        gf = g.reshape(-1, g.shape[-1])
        b = self.cv_beta
        st = self._cv.setdefault(site, dict(S=torch.zeros(gf.shape[1], k.shape[1], device=g.device), num=0., den=0.))
        if 'raw' in st:
            raw = st.pop('raw').reshape(-1, gf.shape[1])
            st['num'] = b * st['num'] + (1 - b) * (gf * raw).sum().item()
            st['den'] = b * st['den'] + (1 - b) * raw.square().sum().item()
        st['S'] = b * st['S'] + (1 - b) * gf.T @ k / k.shape[0]

    def cv_alpha(self, site):
        st = self._cv[site]
        return min(max(st['num'] / st['den'], 0.0), 1.5) if st['den'] > 0 else 0.0

    def output_hook(self, name):
        base = DUST.output_hook(self, name)

        def hook(module, args, output):
            if name in self._frozen and not self.capturing:
                # attention held at its clean output: a residual-stream move then reaches only its own token's loss
                n = output.shape[0] // self.batch
                return self.outputs[name].repeat(n, *([1] * (output.dim() - 1))).to(output.dtype)
            if self.delta_fmt and not self.capturing and isinstance(module, nn.Linear) and name in self.inputs:
                # delta forward: clean output (cached, exact) + Q(W) Q(input - clean input)
                from yoco import quant
                n = args[0].shape[0] // self.batch
                xc = self.inputs[name].repeat(n, *([1] * (args[0].dim() - 1)))
                yc = self.outputs[name].float().repeat(n, *([1] * (output.dim() - 1)))
                key = ('w', name, module.weight._version)
                if self._qw.get(name, (None,))[0] != key:
                    self._qw[name] = (key, quant.quantize_delta(module.weight.detach(), self.delta_fmt)
                                      if self.delta_fmt == 'fp8' else quant.quantize(module.weight.detach()))
                d = quant.quantize_delta(args[0].float() - xc.float(), self.delta_fmt)
                output = (yc + d @ self._qw[name][1].float().t()).to(output.dtype)
            out = base(module, args, output)
            if out is None:
                out = output
            if (name in self._aug or name in self._ro_names) and not self.capturing:
                self._draw_outs[name] = out
            return out
        return hook

    def aug_names(self, layer):
        """Strict K/V errors that a jitter in block `layer` (-1: the embedding) can reach at its own row."""
        out = {}
        for name, e in self.kv_strict.items():
            if self.kv_credit_values_only and not (name.endswith('c_v') or name.endswith('cache_v')):
                continue
            if name.startswith('transformer.cache_'):
                if self.yoco and layer < self.n_self:
                    out[name] = e
            elif int(name.split('.')[2]) > layer:
                out[name] = e
        return out

    def aug_loss(self, n):
        """Predicted loss change through later K/V rows: sum over modules of <e_strict, out - clean_out>."""
        b, t = self.batch, self.length
        tot = 0
        for name, e in self._aug.items():
            d = self._draw_outs[name].float().view(n, b, t, -1) - self.outputs[name].float().view(1, b, t, -1)
            tot = tot + torch.einsum('btw,nbtw->nbt', e, d)
        return tot

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
        if self._anchor_err is not None:
            n = x.shape[0] // self.batch
            with self.autocast():
                xa = self.model(x, None, start=self._start, state=self.rep_state(n), stop_at=self.anchor)
            dx = xa.float().view(n, self.batch, self.length, -1) - self.state['x'][self.anchor].float().unsqueeze(0)
            lin = torch.einsum('nbtd,btd->nbt', dx, self._anchor_err)
            return (self.clean_loss.unsqueeze(0) + lin).reshape(n * self.batch, self.length)
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

    def readout_layers(self, sites):
        """Lower-half blocks whose keys and values a draw at these sites changes, and whether the cache changes."""
        first = 0
        if all(s.startswith('transformer.h.') for s in sites):
            first = min(int(s.split('.')[2]) for s in sites) + 1  # a block's own K/V are computed before its sites
        S = self.n_self if self.yoco else self.model.config.n_layer
        return list(range(first, S)), self.yoco and first <= S

    def readout_names(self, sites):
        layers, cache = self.readout_layers(sites)
        names = {f'transformer.h.{j}.attn.c_{w}' for j in layers for w in 'kv'}
        return names | ({'transformer.cache_k', 'transformer.cache_v'} if cache else set())

    def _readout_prep(self, j):
        """Clean quantities for block j's attention (j = 'cache': every upper-half layer reading the cache)."""
        if j in self._ro_clean:
            return self._ro_clean[j]
        cfg = self.model.config
        b, t, h = self.batch, self.length, cfg.n_head
        d = cfg.n_embd // h
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]  # [s (reader), t (key)]
        out = []
        if j == 'cache':
            k_raw = self.outputs['transformer.cache_k'].float().view(b, t, h, d)
            values = self.outputs['transformer.cache_v'].float().view(b, t, h, d)
            readers = list(range(self.n_self, cfg.n_layer))
        else:
            p = f'transformer.h.{j}.attn.'
            k_raw = self.outputs[p + 'c_k'].float().view(b, t, h, d)
            values = self.outputs[p + 'c_v'].float().view(b, t, h, d)
            if has_ve(j, cfg.n_layer) and f'value_embeds.{j}' in self.outputs:
                z = self.outputs[p + 've_gate'].float()
                values = values + (2 * torch.sigmoid(z)).unsqueeze(-1) * self.outputs[f'value_embeds.{j}'].float().view(b, t, h, d)
            readers = [j]
        kp = self._processed(k_raw, t).float()
        for i in readers:
            qp = self._processed(self.outputs[f'transformer.h.{i}.attn.c_q'].float().view(b, t, h, d), t).float()
            window = self.model.window_sizes[i][0]
            mask = (lag >= 0) & (lag <= window)
            P = (torch.einsum('bshd,bthd->bhst', qp, kp) / d ** 0.5).masked_fill(~mask, float('-inf')).softmax(-1)
            o = torch.einsum('bhst,bthd->bshd', P, values)
            e = self.readout_targets[i].float().view(b, t, h, d)
            ev = torch.einsum('bshd,bthd->bhst', e, values)
            eo = torch.einsum('bshd,bshd->bhs', e, o)
            later = (lag >= 1).float()  # other tokens only: token t's own loss is already in its reward
            out.append((qp, P * later, e, ev - eo[..., None]))
        self._ro_clean[j] = (kp, values, out)
        return self._ro_clean[j]

    def readout_loss(self, n):
        """Predicted change of the other tokens' losses from each token's own key and value change in this draw.

        For key t, the draw's change of its processed key moves one score per reader s, and its value change moves
        one value; a softmax with one changed row has the closed form
            o'_s - o_s = (r (v_t - o_s) + P_st e^delta dv_t) / (1 + r),  r = P_st (e^delta - 1), delta = q_s . dk_t / sqrt(d)
        Scored against reader s's attention-output error e_s and summed over later readers s and all heads: [n, b, t].
        """
        cfg = self.model.config
        b, t, h = self.batch, self.length, cfg.n_head
        d = cfg.n_embd // h
        tot = 0
        layers, cache = self.readout_layers([k for k in self.jitter] or ['transformer.wte'])
        for j in layers + (['cache'] if cache else []):
            kname, vname = ('transformer.cache_k', 'transformer.cache_v') if j == 'cache' else \
                (f'transformer.h.{j}.attn.c_k', f'transformer.h.{j}.attn.c_v')
            if kname not in self._draw_outs:
                continue
            kp, values, readers = self._readout_prep(j)
            k2 = self._draw_outs[kname].float().view(n * b, t, h, d)
            dk = self._processed(k2, t).float().view(n, b, t, h, d) - kp.unsqueeze(0)
            dv = self._draw_outs[vname].float().view(n, b, t, h, d) - self.outputs[vname].float().view(1, b, t, h, d)
            for qp, P, e, ev in readers:
                delta = torch.einsum('bshd,nbthd->nbhst', qp, dk) / d ** 0.5
                ex = torch.exp(delta)
                r = P * (ex - 1)
                edv = torch.einsum('bshd,nbthd->nbhst', e, dv)
                X = (r * ev + P * ex * edv) / (1 + r)  # [n, b, h, s (reader), t (key)]
                if self.readout_mode in ('credit', 'both'):
                    tot = tot + X.sum((2, 3))
                if self.readout_mode in ('leak', 'both'):
                    tot = tot - X.sum((2, 4))
        return tot

    def frozen_for(self, sites):
        """Attention outputs (and YOCO's cache) a draw at these sites should hold clean, for freeze_attn: every
        attention output from the first perturbed block on, except where the draw perturbs that attention itself."""
        L = self.model.config.n_layer
        first = self.start_for(sites) if self.truncate else 0
        if any(not (s.startswith('transformer.h.') or s.startswith('o@')) for s in sites):
            first = 0  # the embedding feeds block 0's attention
        out = {f'transformer.h.{j}.attn.c_proj' for j in range(first, L)
               if f'transformer.h.{j}.attn.c_proj' not in sites and f'o@{j}' not in sites}
        if self.yoco and first < self.n_self:
            out |= {'transformer.cache_k', 'transformer.cache_v'}
        return out

    def direct_errors(self, sites, draws, sigma):  # DUST.direct_errors plus optional residual credit
        """One-sided loss drops, centered across each chunk of draws. """
        self._frozen = self.frozen_for(sites) if self.freeze_attn and self.score is None else set()
        self._ro_names = self.readout_names(sites) if self.readout and self.readout_targets is not None else set()
        try:
            return self._direct_errors(sites, draws, sigma)
        finally:
            self._frozen, self._ro_names = set(), set()
            for k in [k for k in self._draw_outs if k not in self._aug]:
                del self._draw_outs[k]

    def _direct_errors(self, sites, draws, sigma):
        self._start = self.start_for(sites) if self.score is None else 0
        n = self.settings.nrep
        x, y = self.x.repeat(n, 1), self.y.repeat(n, 1)
        width = lambda s: self.model.config.n_embd if s.startswith('o@') else self.outputs[s].shape[-1]
        errors = {s: torch.zeros(self.batch, self.length, width(s), device=self.device) for s in sites}
        halves, chunk = {}, 0
        masks = {s: self.outputs[s] > 0 for s in sites if self.live_fc and s.endswith('mlp.c_fc')}
        mus, use_cv = {}, self.cv and self.score is None and not self.res_gamma
        if use_cv:
            for s in sites:
                raw = self.cv_predict(s)
                if raw is not None and self._cv_steps > self.cv_warmup:
                    mu = self.cv_alpha(s) * raw
                    mus[s] = mu * masks[s] if s in masks else mu

        for _ in range(draws // n):
            noise = {s: self.noise(n * self.batch, self.length, width(s)) for s in sites}
            for s, mask in masks.items():
                noise[s] = noise[s] * mask.repeat(n, 1, 1)
            self.jitter = {s: sigma * a for s, a in noise.items()}
            try:
                perturbed_loss = self.losses(x, y).view(n, self.batch, self.length)
                if self._aug:
                    perturbed_loss = perturbed_loss + self.aug_loss(n)
                if self._ro_names:
                    perturbed_loss = perturbed_loss + self.readout_scale * self.readout_loss(n)
            finally:
                self.jitter = {}

            reward = self.clean_loss.unsqueeze(0) - perturbed_loss
            if use_cv:
                raw_reward = reward - reward.mean(0, keepdim=True) if n > 1 else reward
                if mus:  # remove the reward the memory predicts: sigma * a . mu per jittered site
                    reward = reward + sigma * sum(
                        torch.einsum('nbtd,btd->nbt', noise[s].float().view(n, self.batch, self.length, -1), mu)
                        for s, mu in mus.items())
            if n > 1:
                reward = reward - reward.mean(0, keepdim=True)
            if use_cv:
                st = self.cv_stats.setdefault(sites[0], [0.0, 0.0])
                st[0] += raw_reward.square().sum().item()
                st[1] += reward.square().sum().item()
            if self.res_gamma and any(self.res_site(s) for s in sites):
                # credit token t's jitter with gamma-discounted later token losses (as DUST does for k/v)
                t = self.length
                pos = torch.arange(t, device=self.device)
                lag = pos[None, :] - pos[:, None]  # [t, u]
                K = (lag >= 0).float() * self.res_gamma ** lag.clamp(min=0).float()
                reward = torch.einsum('nbu,tu->nbt', reward, K)
            for site, a in noise.items():
                contrib = torch.einsum(
                    'nbt,nbtd->btd', reward.to(self.dtype),
                    a.view(n, self.batch, self.length, -1),
                ).float() / sigma
                errors[site] -= contrib
                if self.pool:  # half-sums for unbiased signal and noise moments
                    halves.setdefault(site, [0, 0])[chunk % 2] = halves.get(site, [0, 0])[chunk % 2] - contrib
            chunk += 1

        self._start = 0
        if self.pool:
            for site, (h0, h1) in halves.items():
                self._halves[site] = (h0 * 2 / draws, h1 * 2 / draws)
        out = {s: error / draws + mus[s] if s in mus else error / draws for s, error in errors.items()}
        if use_cv:
            for s, g in out.items():
                self.cv_update(s, g)
        return out

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

    @staticmethod
    def separate_scores(kind, P, eo, values, base, credit, dk=None, dv=None, qp=None, ev=None, eb=None):
        """Leak-free local scores: every token's jitter scored on its own, all T experiments in one T x T pass.

        P [b,h,u,s] clean attention probabilities (masked entries 0), eo [b,u,h,d] o@ error targets,
        values [b,s,h,d], base [b,u,h,d] clean output (= P values), credit [s,u] decay for output u -> jitter s.
        kind 'k': dk [n,b,s,h,d] change of the processed key; one shifted logit per query has a closed form,
            o'_u - o_u = r (v_s - o_u) / (1 + r),  r = P_us (exp(q_u . dk_s / sqrt(dh)) - 1).
        kind 'v' / 've' / 'gate': dv [n,b,s,h,d] change of the value row s; o'_u - o_u = P_us dv_s exactly.
        Returns per-jitter scores [n,b,s,h] = sum_u credit[s,u] * eo_u . (o'_u - o_u).
        """
        if kind == 'k':
            dh = qp.shape[-1]
            delta = torch.einsum('buhd,nbshd->nbhus', qp, dk) / dh ** 0.5
            r = P * torch.expm1(delta)
            if ev is None:  # draw-independent; callers may pass them precomputed
                ev = torch.einsum('buhd,bshd->bhus', eo, values)
                eb = torch.einsum('buhd,buhd->bhu', eo, base)
            S = r / (1 + r) * (ev - eb[..., None])
        else:
            S = P * torch.einsum('buhd,nbshd->nbhus', eo, dv)
        return torch.einsum('nbhus,su->nbsh', S, credit)

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
        strict = {}
        self._strict_credit = {}
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
            self._strict_credit[i] = credit * (lag >= 1).float()
            if self.credit_mode == 'attn':
                credit = self.attn_credit(qp, kp, window)
            clean[i] = q, k, z, ve, qp, kp, values, baseline, credit

        sep, sep_e = {}, {}
        if self.local_separate and kind != 'q':
            for i in layers:
                q, k, z, ve, qp, kp, values, baseline, credit = clean[i]
                window = self.model.window_sizes[i][0]
                pos = torch.arange(t, device=self.device)
                lagus = pos[:, None] - pos[None, :]  # [u, s]
                mask = (lagus >= 0) & (lagus <= window)
                logits = torch.einsum('buhd,bshd->bhus', qp.float(), kp.float()) / d ** 0.5
                P = logits.masked_fill(~mask, float('-inf')).softmax(-1)
                vf = values.float()
                sep[i] = P, vf, torch.einsum('bhus,bshd->buhd', P, vf)

        def processed_f(raw):
            return norm(apply_rotary_emb(raw, cos, sin)).float()

        if self.linear_v and kind in ('v', 've'):
            # Values enter attention linearly: e_v[s] = sum_u credit(s, u) p[u, s] e_o[u], from forward quantities.
            for i in layers:
                q, k, z, ve, qp, kp, values, baseline, credit = clean[i]
                window = self.model.window_sizes[i][0]
                pos = torch.arange(t, device=self.device)
                lagus = pos[:, None] - pos[None, :]  # [u, s]
                mask = (lagus >= 0) & (lagus <= window)
                P = (qp.float().transpose(1, 2) @ kp.float().transpose(1, 2).transpose(-1, -2) / d ** 0.5)
                P = P.masked_fill(~mask, float('-inf')).softmax(-1)                      # [b, h, u, s]
                G = torch.where(mask, self.settings.gamma ** lagus.clamp(min=0).float(), torch.zeros((), device=self.device))
                eo = targets[i].float().view(b, t, h, d).transpose(1, 2)                  # [b, h, u, d]
                ev = ((P * G).transpose(-1, -2) @ eo).transpose(1, 2)                     # [b, s, h, d]
                if self.kv_credit and kind == 'v':
                    evs = ((P * G * (lagus >= 1).float()).transpose(-1, -2) @ eo).transpose(1, 2)
                    self.kv_strict[sites[i]] = evs.reshape(b, t, -1)
                if kind == 've':
                    ev = ev * (2 * torch.sigmoid(z.float())).unsqueeze(-1)
                errors[i] = ev.reshape(b, t, -1) * draws
            draws_eff = draws
        for _ in range(0 if (self.linear_v and kind in ('v', 've')) else draws // n):
            for i in layers:
                a = self.noise(n, b, t, width, dtype=torch.float32)
                q, k, z, ve, qp, kp, values, baseline, credit = clean[i]
                window = self.model.window_sizes[i][0]
                divisor = sigma
                if self.local_separate and kind != 'q':
                    assert self.credit_mode == 'gamma' and not self.kv_credit
                    P, vf, of = sep[i]
                    eo = targets[i].float().view(b, t, h, d)
                    if kind == 'k':
                        dk = (processed_f((k.unsqueeze(0) + sigma * a.view(n, b, t, h, d)).reshape(n * b, t, h, d))
                              .view(n, b, t, h, d) - processed_f(k)[None])
                        if i not in sep_e:
                            sep_e[i] = (torch.einsum('buhd,bshd->bhus', eo, vf), torch.einsum('buhd,buhd->bhu', eo, of))
                        score = self.separate_scores('k', P, eo, vf, of, credit, dk=dk, qp=qp.float(),
                                                     ev=sep_e[i][0], eb=sep_e[i][1])
                    else:
                        if kind == 'gate':
                            dv = ((2 * torch.sigmoid(z + sigma * a) - 2 * torch.sigmoid(z - sigma * a)).unsqueeze(-1)
                                  * ve)
                            divisor = 2 * sigma
                        else:
                            dv = sigma * a.view(n, b, t, h, d)
                            if kind == 've':
                                dv = dv * (2 * torch.sigmoid(z)).unsqueeze(-1)
                        score = self.separate_scores(kind, P, eo, vf, of, credit, dv=dv.float())
                    estimate = torch.einsum('nbth,nbthd->bthd', score, a.view(n, b, t, h, -1))
                    errors[i] += estimate.reshape(b, t, width) / divisor
                    continue
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
                if self.kv_credit and kind in ('k', 'v', 'gate'):
                    # credit from strictly later queries only: the extra reward for upstream draws
                    sc = torch.einsum('nbsh,ts->nbth', score.float(), self._strict_credit[i])
                    est_s = torch.einsum('nbth,nbthd->bthd', sc, a.view(n, b, t, h, -1))
                    strict[i] = strict.get(i, 0) + est_s.reshape(b, t, width) / divisor
                if kind != 'q' and self.credit_mode == 'attn':
                    score = torch.einsum('nbuh,bhut->nbth', score.float(), credit)
                elif kind != 'q':
                    score = torch.einsum('nbsh,ts->nbth', score.float(), credit)
                estimate = torch.einsum('nbth,nbthd->bthd', score.float(), a.view(n, b, t, h, -1))
                errors[i] += estimate.reshape(b, t, width) / divisor

        for i, e in strict.items():
            self.kv_strict[sites[i]] = e / draws
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
        cg = self.cache_gamma or self.settings.gamma
        credit = (lag >= 0).float() * cg ** lag.clamp(min=0)
        W = {i: self.attn_credit(qp[i], kp, t) for i in cross} if self.credit_mode == 'attn' and kind != 'q' else None
        if kind == 'v' and self.linear_v:
            G = (lag.T >= 0).float() * cg ** lag.T.clamp(min=0).float()   # [u, s]
            ev, evs = 0, 0
            for i in cross:
                P = (qp[i].float().transpose(1, 2) @ kp.float().transpose(1, 2).transpose(-1, -2) / d ** 0.5)
                P = P.masked_fill(G == 0, float('-inf')).softmax(-1)
                eo = targets[i].float().view(b, t, h, d).transpose(1, 2)
                ev = ev + ((P * G).transpose(-1, -2) @ eo).transpose(1, 2)
                evs = evs + ((P * G * (lag.T >= 1).float()).transpose(-1, -2) @ eo).transpose(1, 2)
            if self.kv_credit:
                self.kv_strict['transformer.cache_v'] = evs.reshape(b, t, -1)
            return {'transformer.cache_v': ev.reshape(b, t, -1)}
        if kind != 'q' and self.cache_separate:
            # Leak-free: each cache row's jitter scored alone against every cross layer (closed form per layer).
            assert self.credit_mode == 'gamma' and not self.kv_credit
            vf = vr.float()
            causal = lag.T >= 0  # [u, s]
            layer = {}
            for i in cross:
                logits = torch.einsum('buhd,bshd->bhus', qp[i].float(), kp.float()) / d ** 0.5
                P = logits.masked_fill(~causal, float('-inf')).softmax(-1)
                of = torch.einsum('bhus,bshd->buhd', P, vf)
                eo = targets[i].float().view(b, t, h, d)
                extra = (torch.einsum('buhd,bshd->bhus', eo, vf), torch.einsum('buhd,buhd->bhu', eo, of)) \
                    if kind == 'k' else (None, None)
                layer[i] = P, eo, of, extra
            kf = self._processed(kr, t).float()
            err = torch.zeros(b, t, cfg.n_embd, device=self.device)
            for _ in range(draws // n):
                a = self.noise(n, b, t, h * d, dtype=torch.float32)
                if kind == 'k':
                    dk = (self._processed((kr.unsqueeze(0) + sigma * a.view(n, b, t, h, d)).reshape(n * b, t, h, d), t)
                          .float().view(n, b, t, h, d) - kf[None])
                score = 0
                for i in cross:
                    P, eo, of, (ev, eb) = layer[i]
                    if kind == 'k':
                        score = score + self.separate_scores('k', P, eo, vf, of, credit, dk=dk, qp=qp[i].float(),
                                                             ev=ev, eb=eb)
                    else:
                        score = score + self.separate_scores('v', P, eo, vf, of, credit,
                                                             dv=sigma * a.view(n, b, t, h, d))
                err += torch.einsum('nbth,nbthd->bthd', score, a.view(n, b, t, h, d)).reshape(b, t, -1) / sigma
            return {f'transformer.cache_{kind}': err / draws}
        if kind == 'q':
            errors = {i: torch.zeros(b, t, cfg.n_embd, device=self.device) for i in cross}
        else:
            err = torch.zeros(b, t, cfg.n_embd, device=self.device)
            err_strict = 0
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
            if self.kv_credit and self.credit_mode != 'attn':
                sc = torch.einsum('nbsh,ts->nbth', score, credit * (lag >= 1).float())
                err_strict = err_strict + torch.einsum('nbth,nbthd->bthd', sc, a.view(n, b, t, h, d)).reshape(b, t, -1) / sigma
            score = credited if self.credit_mode == 'attn' else torch.einsum('nbsh,ts->nbth', score, credit)
            err += torch.einsum('nbth,nbthd->bthd', score, a.view(n, b, t, h, d)).reshape(b, t, -1) / sigma
        if kind == 'q':
            return {f'transformer.h.{i}.attn.c_q': e / draws for i, e in errors.items()}
        if self.kv_credit:
            self.kv_strict[f'transformer.cache_{kind}'] = err_strict / draws
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
        hit = lambda n: self.group(n) in self.oracle or (self.oracle_re is not None and re.search(self.oracle_re, n))
        params = {n: p for n, p in m.named_parameters() if hit(n)}
        fast_head = m.lm_head.__dict__.pop('forward', None)  # fast16's fp32-output mm has no autograd rule
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
            if fast_head is not None:
                m.lm_head.forward = fast_head
        return {n: g.float() for n, g in zip(params, gs)}

    # ---- local chaining: cheap local probes scored against pooled stream errors ------------------------
    def local_mlp(self, layer, target, draws, sigma=0.2, n=16):
        """Error at mlp.c_fc's output: jitter it, rerun only relu^2 and mlp.c_proj, score the change in the MLP
        output against `target` (the error at mlp.c_proj's output). Antithetic pairs; forward only."""
        p = f'transformer.h.{layer}.mlp.'
        h = self.outputs[p + 'c_fc'].float()
        W2 = self.modules[p + 'c_proj'].weight.float()
        b, t, f = h.shape
        base = torch.relu(h).square()
        est = torch.zeros_like(h)
        for _ in range(draws // n):
            a = torch.randn(n, b, t, f, device=self.device, generator=self.generator)
            dp = (torch.relu(h + sigma * a).square() - base) @ W2.t()
            dm = (torch.relu(h - sigma * a).square() - base) @ W2.t()
            score = torch.einsum('btd,nbtd->nbt', target, (dp - dm) / 2)
            est += torch.einsum('nbt,nbtf->btf', score, a) / sigma
        return est / draws

    def local_proj(self, layer, target, draws, sigma=0.2, n=16):
        """Error at the attention output (o@): jitter it, rerun only attn.c_proj, score against `target`."""
        W = self.modules[f'transformer.h.{layer}.attn.c_proj'].weight.float()
        b, t, d = target.shape
        est = torch.zeros(b, t, W.shape[1], device=self.device)
        for _ in range(draws // n):
            a = torch.randn(n, b, t, W.shape[1], device=self.device, generator=self.generator)
            score = torch.einsum('btd,nbtd->nbt', target, (sigma * a) @ W.t())
            est += torch.einsum('nbt,nbtf->btf', score, a) / sigma
        return est / draws

    # ---- residual pooling --------------------------------------------------------------------------
    def pool_writers(self, errors):
        """Combine every writer site's error estimate into each site's estimate.

        Residual-stream errors at different depths are nearly parallel, so the other sites' draws are extra
        draws for this one. With unbiased moments from independent draws (cross-site products, split-half
        products, both averaged over steps) each site s gets
            pooled_s = sum_j w_j k_sj est_j / sum_j w_j,  k_sj = <g_s, g_j> / |g_j|^2,
            w_j = 1 / (k_sj^2 noise_j + (1 - cos_sj^2) |g_s|^2)     (noise plus the bias of using site j)
        """
        sites = [s for s in self._halves if s in errors and (s.endswith('c_proj') or s == 'transformer.wte')]
        if len(sites) < 2:
            return errors
        S = len(sites)
        C = torch.zeros(S, S, dtype=torch.float64)
        noise = torch.zeros(S, dtype=torch.float64)
        for i, a in enumerate(sites):
            h0, h1 = self._halves[a]
            C[i, i] = (h0 * h1).sum().item()
            noise[i] = ((h0 - h1).square().sum().item()) / 4.0
            for j in range(i + 1, S):
                C[i, j] = C[j, i] = (errors[a] * errors[sites[j]]).sum().item()
        key = tuple(sites)
        if self._pool_ema.get(key) is None:
            self._pool_ema[key] = (C, noise)
        else:
            C0, n0 = self._pool_ema[key]
            self._pool_ema[key] = (self.pool_decay * C0 + (1 - self.pool_decay) * C,
                                   self.pool_decay * n0 + (1 - self.pool_decay) * noise)
        C, noise = self._pool_ema[key]
        diag = C.diag().clamp_min(1e-30)
        out = dict(errors)
        for i, a in enumerate(sites):
            if C[i, i] <= 0:
                continue
            num, den = 0, 0.0
            for j, bname in enumerate(sites):
                if j != i and (C[j, j] <= 0 or C[i, j] <= 0):
                    continue
                k = (C[i, j] / diag[j]).item() if j != i else 1.0
                cos2 = (C[i, j] ** 2 / (diag[i] * diag[j])).clamp(max=1.0).item() if j != i else 1.0
                w = 1.0 / (k * k * noise[j].item() + (1 - cos2) * diag[i].item() + 1e-30)
                num = num + w * k * errors[bname]
                den += w
            out[a] = num / den
        return out

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
                if self.moe_prefixes:  # MoE layer: the routed hidden units and gate logits join the hidden draws
                    sites = sites + [q + suf for q in self.moe_prefixes for s_ in sites if s_ == q + 'c_fc'
                                     for suf in ('moe', 'gate')]
                errors.update(self.direct_errors(sites, count, sigma))

        def hub(layer):
            sigma = 0.2 if layer < L // 2 else 0.4
            return self.direct_errors([f'o@{layer}'], r.hubs[layer], sigma)[f'o@{layer}']

        if self.kv_credit and not self.handoff:
            # Hubs and attention internals first: their strictly-later K/V errors augment every upstream reward.
            self.kv_strict = {}
            targets = {layer: hub(layer) for layer in range(L)}
            if self.yoco:
                for kind in ('q', 'k', 'v'):
                    errors.update(self.zo_cross(kind, r.local[kind], targets))
            for kind, count in r.local.items():
                errors.update(self.zo_self_local(kind, count, targets))
            for layer in range(L):
                p = f'transformer.h.{layer}.'
                self._aug = self.aug_names(layer)
                direct([p + 'attn.c_proj', p + 'mlp.c_proj'], r.writers[layer], 0.2)
                direct([p + 'mlp.c_fc'], r.hidden[layer], 0.2)
            self._aug = self.aug_names(-1) if self.kv_credit_embedding else {}
            direct(['transformer.wte'], r.embedding, 0.2)
            self._aug, self._draw_outs = {}, {}
            return self.finish(x, y, optimizer, errors)

        self._steps += 1
        if self.readout and self._steps > self.readout_warmup and self.chain and not self.chain_pool and self.chain_hidden is False and not self.handoff:
            # Cross-token readout: blocks top-down, so every reader's attention-output target (chained from its
            # block's own c_proj error) exists before a lower block's draws are scored against it.
            targets = {}
            self.readout_targets = targets
            for layer in reversed(range(L)):
                p = f'transformer.h.{layer}.'
                direct([p + 'attn.c_proj', p + 'mlp.c_proj'], r.writers[layer], 0.2)
                direct([p + 'mlp.c_fc'], r.hidden[layer], 0.2)
                targets[layer] = self.local_proj(layer, errors[p + 'attn.c_proj'], self.chain_draws)
            direct(['transformer.wte'], r.embedding, 0.2)
            self._halves, self.readout_targets = {}, None
            if self.yoco:
                for kind in ('q', 'k', 'v'):
                    errors.update(self.zo_cross(kind, r.local[kind], targets))
            for kind, count in r.local.items():
                errors.update(self.zo_self_local(kind, count, targets))
            return self.finish(x, y, optimizer, errors)

        if self.chain and not self.handoff and not self.kv_credit:
            # Full forwards only at residual-stream sites (writers, embedding), pooled across depths; MLP-hidden
            # and attention-output errors come from cheap local probes scored against the pooled stream errors.
            A = self.anchor if self.anchor and not self.chain_hidden and not self.chain_pool else 0
            for layer in range(A, L):
                p = f'transformer.h.{layer}.'
                direct([p + 'attn.c_proj', p + 'mlp.c_proj'], r.writers[layer], 0.2)
                if not self.chain_hidden:
                    direct([p + 'mlp.c_fc'], r.hidden[layer], 0.2)
            if A:
                # block A-1's writers by full draws give the anchor error; everything below reruns only to block A
                p = f'transformer.h.{A - 1}.'
                direct([p + 'attn.c_proj', p + 'mlp.c_proj'], r.writers[A - 1], 0.2)
                self._anchor_err = errors[p + 'mlp.c_proj'].float()
                k = self.anchor_scale
                try:
                    direct([p + 'mlp.c_fc'], k * r.hidden[A - 1], 0.2)
                    for layer in range(A - 1):
                        q = f'transformer.h.{layer}.'
                        direct([q + 'attn.c_proj', q + 'mlp.c_proj'], k * r.writers[layer], 0.2)
                        direct([q + 'mlp.c_fc'], k * r.hidden[layer], 0.2)
                    direct(['transformer.wte'], k * r.embedding, 0.2)
                finally:
                    self._anchor_err = None
            else:
                direct(['transformer.wte'], r.embedding, 0.2)
            if self.chain_pool:
                errors = self.pool_writers(errors)
            self._halves = {}
            targets = {}
            for layer in range(L):
                p = f'transformer.h.{layer}.'
                if self.chain_hidden:
                    errors[p + 'mlp.c_fc'] = self.local_mlp(layer, errors[p + 'mlp.c_proj'], self.chain_draws)
                targets[layer] = self.local_proj(layer, errors[p + 'attn.c_proj'], self.chain_draws)
            if self.yoco:
                for kind in ('q', 'k', 'v'):
                    errors.update(self.zo_cross(kind, r.local[kind], targets))
            for kind, count in r.local.items():
                errors.update(self.zo_self_local(kind, count, targets))
            return self.finish(x, y, optimizer, errors)

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

        return self.finish(x, y, optimizer, errors)

    def finish(self, x, y, optimizer, errors):
        r = self.settings
        grads = {}
        if self.pool:
            errors = self.pool_writers(errors)
            self._halves = {}
        if self.head_mode == 'exact':
            errors['lm_head'] = self.exact_head_error()
        else:
            errors['lm_head'] = self.head_error(r.head)

        for q in self.moe_prefixes:
            grads.update(self.modules_moe[q].dust_grads(self, q, errors.pop(q + 'moe'), errors.pop(q + 'gate'),
                                                        errors[q + 'c_proj']))
        for site, error in errors.items():
            grads[site + '.weight'] = self.reassemble(site, error)
        grads.update(self.scalar_gradients(8))
        if self.oracle or self.oracle_re:
            grads.update(self.expected_grads(x, y))

        valid = (y != -1).sum()
        if self.accum is not None:  # microbatch: sum the raw estimates; the caller divides and steps
            for name, g in grads.items():
                self.accum[name] = self.accum[name] + g if name in self.accum else g
            self.accum_valid += valid
            return self.clean_loss[y != -1].mean().item()
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.model.named_parameters():
            g = (grads[name] / valid).to(parameter.dtype)
            if not torch.isfinite(g).all():
                raise FloatingPointError(f'Non-finite estimate for {name}')
            parameter.grad = g
        optimizer.step()
        return self.clean_loss[y != -1].mean().item()
