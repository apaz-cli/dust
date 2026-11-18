"""Fine-grained routed MoE MLP for the small YOCO/GPT models (DeepSeekMoE-style, scaled down).

Each MoE layer has E routed experts (relu^2 MLPs with hidden size h), top-k routing on sigmoid scores with the
selected weights normalised to sum to 1, an optional shared expert (hidden size hs) that every token uses, and
DeepSeek's auxiliary-loss-free load balancing: a per-expert bias added to the scores for selection only, nudged
after every step toward equal load. The bias update needs no gradient, so it fits a ZO trainer as is.

Active hidden width per token is k*h + hs; a dense MLP with hidden 4d matches it when k*h + hs = 4d.
Routing is dropless: every token goes to its top-k experts.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class MoE(nn.Module):
    def __init__(self, d, experts, topk, hidden, shared_hidden=0, balance_rate=1e-3):
        super().__init__()
        self.E, self.k, self.h, self.hs, self.rate = experts, topk, hidden, shared_hidden, balance_rate
        self.router = nn.Linear(d, experts, bias=False)
        self.w_fc = nn.Parameter(torch.empty(experts, hidden, d))
        self.w_proj = nn.Parameter(torch.empty(experts, d, hidden))
        if shared_hidden:
            self.c_fc = nn.Linear(d, shared_hidden, bias=False)
            self.c_proj = nn.Linear(shared_hidden, d, bias=False)
        self.register_buffer('balance_bias', torch.zeros(experts))
        self.register_buffer('load', torch.zeros(experts))  # tokens routed per expert, accumulated since last update
        self.dust = None
        self.cap = 0.0        # soft cap on router logits (0: off)
        self.dense = False    # compute every expert for every token, keep the routed ones: fixed shapes, compilable
        self.grouped = False  # routed experts by one grouped matmul over the slots sorted by expert: fixed shapes, compilable
        self.logit_rms = 0.0  # size of the router logits in the last clean pass, for logging

    def init_weights(self):
        d = self.w_fc.shape[-1]
        s = 3 ** 0.5 * d ** -0.5
        nn.init.uniform_(self.w_fc, -s, s)
        nn.init.uniform_(self.w_proj, -1.5 * math.sqrt(3 / self.h), 1.5 * math.sqrt(3 / self.h))
        nn.init.uniform_(self.router.weight, -s, s)
        if self.hs:
            nn.init.uniform_(self.c_fc.weight, -s, s)
            nn.init.uniform_(self.c_proj.weight, -1.5 * math.sqrt(3 / self.hs), 1.5 * math.sqrt(3 / self.hs))
        self.balance_bias.zero_()
        self.load.zero_()

    def forward(self, x):
        """x [B, T, d]. Under a Dust estimator (self.dust = (est, prefix)), the clean pass records the routing,
        the routed pre-activations (site prefix+'moe', width k*h) and the selected router logits (site
        prefix+'gate', width k); draws reuse the clean routing and add the estimator's jitter at those sites."""
        B, T, d = x.shape
        raw = self.router(x).float()                                 # [B, T, E]
        capf = (lambda z: self.cap * torch.tanh(z / self.cap)) if self.cap else (lambda z: z)
        logits = capf(raw)
        est, prefix = self.dust if self.dust is not None else (None, None)
        if est is None or est.capturing:
            self.logit_rms = raw.detach().square().mean().sqrt()  # a tensor: read when logging, not per forward
        if est is not None and (prefix + 'moe' in est.jitter or prefix + 'gate' in est.jitter
                                or (getattr(est, 'pin_routing', False) and not est.capturing)):
            ids = est.outputs[prefix + 'ids']  # draws at this layer's own sites keep the clean routing
            ids = ids.repeat(B // ids.shape[0], 1, 1)
        else:
            ids = (torch.sigmoid(logits) + self.balance_bias).topk(self.k, dim=-1).indices
            if est is None or est.capturing or not est.jitter:
                self.load += self._counts(ids).float()
        self.last_ids = ids
        sel = raw.gather(-1, ids)                                     # [B, T, k], before the cap: the gate site
        if est is not None and prefix + 'gate' in est.jitter:
            sel = sel + est.jitter[prefix + 'gate'].float()
        w = torch.sigmoid(capf(sel))
        w = w / w.sum(-1, keepdim=True)
        xf = x.reshape(-1, d)
        N = xf.shape[0]
        if self.dense:
            # every expert on every token (E/k times the expert FLOPs), then keep each token's k routed slots: no sort,
            # no padding, no host sync, fixed shapes. Same values as the sparse path below.
            idx = ids.reshape(N, self.k, 1).expand(N, self.k, self.h)
            pre = torch.einsum('nd,ehd->neh', xf, self.w_fc.to(x.dtype)).float().gather(1, idx).reshape(N * self.k, self.h)
            if est is not None and prefix + 'moe' in est.jitter:
                pre = pre + est.jitter[prefix + 'moe'].reshape(N * self.k, self.h).to(pre.dtype)
            hid = F.relu(pre).square()
            if est is not None and est.capturing:
                est.outputs[prefix + 'ids'] = ids
                est.outputs[prefix + 'moe'] = pre.view(B, T, self.k * self.h)
                est.outputs[prefix + 'gate'] = sel
                est.inputs[prefix + 'moe'] = xf.float()
                est.inputs[prefix + 'hid'] = hid.float().view(N, self.k, self.h)
                est.inputs[prefix + 'w'] = w.reshape(N, self.k)
            H = torch.zeros(N, self.E, self.h, device=x.device, dtype=torch.float32)
            H.scatter_(1, idx, hid.view(N, self.k, self.h) * w.reshape(N, self.k, 1))
            out = torch.einsum('neh,edh->nd', H.to(x.dtype), self.w_proj.to(x.dtype)).float().view(B, T, d)
        elif self.grouped and x.is_cuda and not (torch.is_grad_enabled() and self.w_fc.requires_grad):
            out = self._grouped(x, xf, N, ids, w, sel, est, prefix).view(B, T, d)
        else:
            out = self._sparse(x, xf, N, ids, w, sel, est, prefix).view(B, T, d)
        if self.hs:
            out = out + self.c_proj(F.relu(self.c_fc(x)).square()).float()
        return out.to(x.dtype)


    def _counts(self, ids):
        """Tokens per expert, [E] int64; a fixed-size output (bincount's size depends on the data, which breaks compile)."""
        flat = ids.reshape(-1)
        return torch.zeros(self.E, dtype=torch.int64, device=ids.device).scatter_add_(0, flat, torch.ones_like(flat))

    @staticmethod
    def _mm_dtype(x):
        return torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype

    def _grouped(self, x, xf, N, ids, w, sel, est, prefix):
        """Routed experts as in _sparse, but with one grouped matmul per projection over the N*k token slots sorted by
        expert (yoco/gmm.py, a Triton kernel with the per-expert offsets on the device): no padding, no host sync,
        fixed shapes, so the forward compiles. Inputs in the autocast dtype (fp16 under fast16), fp32 accumulation and
        output. Forward only: training with gradients (backprop) takes _sparse."""
        from yoco.gmm import gmm
        B, T, d = x.shape
        dt = self._mm_dtype(x)
        flat_ids = ids.reshape(-1)                                    # [N*k], slot-major per token
        order = flat_ids.argsort(stable=True)
        inv = order.argsort()
        tok = order // self.k                                         # the token of each sorted slot
        offs = torch.cumsum(self._counts(ids), 0).to(torch.int32)     # end of each expert's group
        pre = gmm(xf[tok].to(dt), self.w_fc.to(dt).transpose(1, 2), offs)[inv]
        if est is not None and prefix + 'moe' in est.jitter:
            pre = pre + est.jitter[prefix + 'moe'].reshape(N * self.k, self.h).to(pre.dtype)
        hid = F.relu(pre).square()
        if est is not None and est.capturing:
            est.outputs[prefix + 'ids'] = ids
            est.outputs[prefix + 'moe'] = pre.view(B, T, self.k * self.h)
            est.outputs[prefix + 'gate'] = sel
            est.inputs[prefix + 'moe'] = xf.float()
            est.inputs[prefix + 'hid'] = hid.float().view(N, self.k, self.h)
            est.inputs[prefix + 'w'] = w.reshape(N, self.k)
        Y = gmm(hid[order].to(dt), self.w_proj.to(dt).transpose(1, 2), offs)          # [N*k, d], sorted order
        out = torch.zeros(N, d, device=x.device, dtype=torch.float32)
        out.index_add_(0, tok, Y * w.reshape(-1)[order, None])
        return out

    def _sparse(self, x, xf, N, ids, w, sel, est, prefix):
        """Routed experts on padded per-expert batches (sorts the tokens; the largest batch is read back to the host)."""
        B, T, d = x.shape
        flat_ids = ids.reshape(-1)                                    # [N*k], slot-major per token
        tok = torch.arange(N, device=x.device).repeat_interleave(self.k)
        # padded per-expert batches: slot i of the sorted list goes to (expert, rank within that expert)
        order = flat_ids.argsort()
        eid = flat_ids[order]
        counts = torch.bincount(flat_ids, minlength=self.E)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(N * self.k, device=x.device) - starts[eid]
        cap = int(counts.max())
        Xp = torch.zeros(self.E, cap, d, device=x.device, dtype=x.dtype)
        Xp[eid, rank] = xf[tok[order]]
        pre = torch.empty(N * self.k, self.h, device=x.device, dtype=torch.float32)
        pre[order] = torch.bmm(Xp, self.w_fc.to(x.dtype).transpose(1, 2))[eid, rank].float()
        if est is not None and prefix + 'moe' in est.jitter:
            pre = pre + est.jitter[prefix + 'moe'].reshape(N * self.k, self.h).to(pre.dtype)
        hid = F.relu(pre).square()
        if est is not None and est.capturing:
            est.outputs[prefix + 'ids'] = ids
            est.outputs[prefix + 'moe'] = pre.view(B, T, self.k * self.h)
            est.outputs[prefix + 'gate'] = sel
            est.inputs[prefix + 'moe'] = xf.float()
            est.inputs[prefix + 'hid'] = hid.float().view(N, self.k, self.h)
            est.inputs[prefix + 'w'] = w.reshape(N, self.k)
        Hp = torch.zeros(self.E, cap, self.h, device=x.device, dtype=x.dtype)
        Hp[eid, rank] = hid[order].to(x.dtype)
        Y = torch.bmm(Hp, self.w_proj.to(x.dtype).transpose(1, 2))[eid, rank].float()   # [N*k, d] in sorted order
        out = torch.zeros(N, d, device=x.device, dtype=torch.float32)
        out.index_add_(0, tok[order], Y * w.reshape(-1)[order, None])
        return out

    @torch.no_grad()
    def local_expert_errors(self, est, prefix, e_out, draws, sigma=0.2, n=8):
        """Error at the routed pre-activations (site prefix+'moe') from local probes instead of full reruns: jitter
        each token's live expert units, rerun only relu^2 and that expert's output projection (times its gate weight),
        and score the change of the layer output against e_out, the error there (estimated from all tokens). Antithetic
        pairs; forward only. Returns [b, t, k*h]."""
        pre = est.outputs[prefix + 'moe'].float()                          # [b, t, k*h]
        b, t, _ = pre.shape
        N = b * t
        ids = est.outputs[prefix + 'ids'].reshape(-1)                      # [N*k]
        w = est.inputs[prefix + 'w'].reshape(-1)                           # [N*k]
        pre = pre.reshape(N * self.k, self.h)
        live = (pre > 0).float()
        base = F.relu(pre).square()
        tok = torch.arange(N, device=pre.device).repeat_interleave(self.k)
        order = ids.argsort()
        eid = ids[order]
        counts = torch.bincount(ids, minlength=self.E)
        rank = torch.arange(N * self.k, device=pre.device) - (torch.cumsum(counts, 0) - counts)[eid]
        cap = int(counts.max())
        Wp = self.w_proj.float()
        eo = e_out.reshape(N, -1)[tok] * w[:, None]                        # [N*k, d]: each slot's output error, gated
        est_ = torch.zeros_like(pre)

        def project(dh):  # [m, N*k, h] -> [m, N*k, d] through each slot's expert
            m = dh.shape[0]
            Hp = torch.zeros(m, self.E, cap, self.h, device=pre.device)
            Hp[:, eid, rank] = dh[:, order]
            Y = torch.einsum('mech,edh->mecd', Hp, Wp)[:, eid, rank]
            out = torch.empty(m, N * self.k, Y.shape[-1], device=pre.device)
            out[:, order] = Y
            return out
        for _ in range(draws // n):
            a = torch.randn(n, N * self.k, self.h, device=pre.device, generator=est.generator) * live
            dp = project(F.relu(pre + sigma * a).square() - base)
            dm = project(F.relu(pre - sigma * a).square() - base)
            score = ((dp - dm) / 2 * eo).sum(-1)                           # [n, N*k]
            est_ += (score.unsqueeze(-1) * a).sum(0) / sigma
        return (est_ / draws).reshape(b, t, self.k * self.h)

    def dust_grads(self, est, prefix, e_pre, e_gate, e_out):
        """Weight estimates from the site errors: e_pre [b,t,k*h] at the routed pre-activations, e_gate [b,t,k] at
        the selected router logits, e_out [b,t,d] at the layer output (the shared expert's c_proj site)."""
        x, hid, w = est.inputs[prefix + 'moe'], est.inputs[prefix + 'hid'], est.inputs[prefix + 'w']
        ids = est.outputs[prefix + 'ids'].reshape(-1, self.k)
        N = x.shape[0]
        e_pre = e_pre.reshape(N, self.k, self.h)
        e_out = e_out.reshape(N, -1)
        g_fc = torch.zeros_like(self.w_fc, dtype=torch.float32)
        g_proj = torch.zeros_like(self.w_proj, dtype=torch.float32)
        for e in range(self.E):
            n_, j_ = (ids == e).nonzero(as_tuple=True)
            if len(n_):
                g_fc[e] = e_pre[n_, j_].t() @ x[n_]
                g_proj[e] = (e_out[n_] * w[n_, j_, None]).t() @ hid[n_, j_]
        g_router = torch.zeros(self.E, x.shape[1], device=x.device)
        g_router.index_add_(0, ids.reshape(-1), (e_gate.reshape(N, self.k, 1) * x[:, None]).reshape(-1, x.shape[1]))
        return {prefix + 'w_fc': g_fc, prefix + 'w_proj': g_proj, prefix + 'router.weight': g_router}

    @torch.no_grad()
    def update_balance(self):
        """Aux-loss-free balancing (Wang et al. 2024): raise the bias of under-loaded experts, lower over-loaded."""
        if self.load.sum() > 0:
            self.balance_bias += self.rate * torch.sign(self.load.mean() - self.load)
        self.load.zero_()


def install(model, experts, topk, hidden, shared_hidden=0, layers=None):
    """Replace block.mlp with an MoE in the chosen layers (default all). Returns the MoE modules."""
    d = model.config.n_embd
    dev = next(model.parameters()).device
    out = []
    for i, blk in enumerate(model.transformer.h):
        if layers is not None and i not in layers:
            continue
        m = MoE(d, experts, topk, hidden, shared_hidden).to(dev)
        m.init_weights()
        blk.mlp = m
        out.append(m)
    return out


def balance(model):
    for m in model.modules():
        if isinstance(m, MoE):
            m.update_balance()


def load_stats(model):
    """Max/mean load ratio per MoE layer since the last balance call (1.0 = perfectly balanced)."""
    return [round((m.load.max() / m.load.mean().clamp(min=1)).item(), 2) for m in model.modules() if isinstance(m, MoE)]
