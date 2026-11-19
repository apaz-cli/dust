"""Count the floating-point operations of one training step (Dust or backprop), measured, not estimated.

`count_step(run_step, model, tokens)` runs one step under torch's FlopCounterMode and returns the total, the count
per training token, and a breakdown by module. FlopCounterMode sees ATen operations only, so the step runs uncompiled
(the caller switches compilation off for it), and the grouped expert matmul (dust::gmm, a Triton kernel) gets its
formula registered here. Matrix multiplies count 2 * M * N * K; attention (scaled_dot_product_attention) is counted
as a full T x T product even when causal, so it is an upper bound there (a causal kernel does about half).
`forward_flops(model, x, y)` counts one plain forward pass, the unit Dust's cost is usually quoted in.
"""
import torch
from torch.utils.flop_counter import FlopCounterMode, register_flop_formula

try:
    from yoco import gmm as _gmm  # noqa: F401  (registers dust::gmm)

    @register_flop_formula(torch.ops.dust.gmm)
    def _gmm_flop(a_shape, b_shape, offs_shape, *args, out_shape=None, **kwargs):
        M, K = a_shape
        return 2 * M * K * b_shape[-1]
except Exception:  # no triton (CPU-only machine): the MoE grouped path is not used there either
    pass


def _breakdown(counter, depth_prefix='transformer.h.'):
    """Total per module at the level of blocks and their parts (attn, mlp) plus the head and embedding."""
    out = {}
    for mod, ops in counter.get_flop_counts().items():
        if mod == 'Global':
            continue
        parts = mod.split('.')
        if parts[0] not in ('transformer', 'lm_head', 'value_embeds'):  # the root module's class name comes first
            parts = parts[1:]
        if not parts:
            continue
        name = '.'.join(parts)
        # keep e.g. transformer.h.3.attn, transformer.h.3.mlp, lm_head, transformer.wte
        if name.startswith(depth_prefix) and len(parts) == 4:
            out[name] = out.get(name, 0) + sum(ops.values())
        elif len(parts) <= 2 and not name.startswith(depth_prefix):
            out[name] = out.get(name, 0) + sum(ops.values())
    return out


def count_step(run_step, model, tokens):
    """run_step(): performs one training step. tokens: training tokens in that step."""
    counter = FlopCounterMode(display=False, depth=4)
    with counter:
        run_step()
    total = counter.get_total_flops()
    return dict(total=total, per_token=total / tokens, breakdown=_breakdown(counter))


@torch.no_grad()
def forward_flops(model, x, y):
    counter = FlopCounterMode(display=False, depth=1)
    with counter, torch.autocast('cuda', dtype=torch.bfloat16):  # as in evaluation; the count does not depend on it
        model(x, y)
    return counter.get_total_flops()
