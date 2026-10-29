"""YOCO variant of the DUST transformer: a self-decoder, one shared KV cache, then a cross-decoder.

The first n_self blocks are ordinary windowed self-attention blocks. Their output X (the handoff) is
normalized once and projected by cache_k / cache_v into a single KV cache. Every later block attends
only to that cache with its own queries, so a cross-decoder activation at token t reaches no other
token: in the cross-decoder, DUST's per-token reward is exact and leak-free.

`transformer.handoff` is an identity on the residual copy of X. Jittering it moves the residual
stream entering the cross-decoder while the cache stays clean.
"""
import math

import torch
from torch import nn

from model import GPT, GPTConfig, MLP, Block, attention, apply_rotary_emb, has_ve, norm


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate = None

    def forward(self, x, k, v, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q = norm(apply_rotary_emb(q, cos, sin))
        y = attention(q, k, v, window_size[0])
        return self.c_proj(y.contiguous().view(B, T, -1))


class CrossBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CrossAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, k, v, cos_sin, window_size):
        x = x + self.attn(norm(x), k, v, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class YOCO(GPT):
    def __init__(self, config, n_self=None, self_window=None):
        super().__init__(config)
        self.n_self = n_self if n_self is not None else config.n_layer // 2
        L, T = config.n_layer, config.sequence_len
        w = self_window if self_window is not None else T // 2
        self.window_sizes = [(w, 0)] * self.n_self + [(T, 0)] * (L - self.n_self)
        self.transformer.h = nn.ModuleList(
            [Block(config, i) for i in range(self.n_self)] + [CrossBlock(config) for _ in range(self.n_self, L)]
        )
        kv_dim = config.n_kv_head * (config.n_embd // config.n_head)
        self.transformer.cache_k = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.transformer.cache_v = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.transformer.handoff = nn.Identity()
        # Value embeddings only where a block computes its own values: the self-decoder.
        self.value_embeds = nn.ModuleDict({k: e for k, e in self.value_embeds.items() if int(k) < self.n_self})
        for i in range(self.n_self, L):
            self.transformer.h[i].attn.ve_gate = None

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3 ** 0.5 * self.config.n_embd ** (-0.5)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name != 'lm_head':
                torch.nn.init.uniform_(module.weight, -s, s)
            if name.endswith('ve_gate') and isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        (self.cos, self.sin) = (cos, sin)
        if self.transformer.wte.weight.device.type == 'cuda':
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def cache(self, x, cos_sin):
        B, T, _ = x.shape
        h, d = self.config.n_kv_head, self.config.n_embd // self.config.n_head
        xn = norm(x)
        k = self.transformer.cache_k(xn).view(B, T, h, d)
        v = self.transformer.cache_v(xn).view(B, T, h, d)
        cos, sin = cos_sin
        return norm(apply_rotary_emb(k, cos, sin)), v

    def forward(self, idx, targets=None, loss_reduction='mean', stop_at_handoff=False):
        B, T = idx.size()
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i in range(self.n_self):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
        if stop_at_handoff:
            return x
        k, v = self.cache(x, cos_sin)
        x = self.transformer.handoff(x)
        for i in range(self.n_self, self.config.n_layer):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = self.transformer.h[i](x, k, v, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is not None:
            return nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
        return logits


def build(arch, config, device, seed, n_self=None):
    """make_model for either architecture: output projections at std 1.5 / sqrt(fan_in), as dust.make_model."""
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config) if arch == 'gpt' else YOCO(config, n_self=n_self)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = 1.5 * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
    return model.requires_grad_(False)
