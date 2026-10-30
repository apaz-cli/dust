"""Simulated NVFP4 weights for DUST forwards.

NVFP4: blocks of 16 consecutive input weights share an e4m3 scale; each value is e2m1, one of
{0, 0.5, 1, 1.5, 2, 3, 4, 6} with a sign; a per-tensor fp32 scale keeps the block scales in e4m3 range.
Weights are fake-quantized (quantize, then dequantize for an fp16 matmul): the forward computes what an NVFP4
model computes. The clean and jittered passes see the same quantized weights, so quantization adds no noise to
DUST's rewards.

  master: forwards use Q(W) of fp32 master weights, updates go to the master (quantization-aware training).
  sr:     no master: after every update the weights are re-quantized with stochastic rounding and stored on
          the NVFP4 grid, as a 4-bit-resident model would be.
"""
import types

import torch
from torch import nn
from torch.nn import functional as F

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quantize(w, stochastic=False, block=16):
    """Fake-quantize a 2-D weight to NVFP4 along its input dimension."""
    shape = w.shape
    wf = w.float().reshape(-1, block)
    g = wf.abs().max() / (6.0 * 448.0)                       # per-tensor scale: block scales fit e4m3 (max 448)
    g = torch.where(g > 0, g, torch.ones_like(g))
    s = (wf.abs().amax(-1, keepdim=True) / 6.0 / g)          # block scale in units of g
    s = s.to(torch.float8_e4m3fn).float().clamp_min(2 ** -9)
    v = wf / (s * g)                                         # now in [-6, 6]
    grid = E2M1.to(w.device)
    a = v.abs().clamp(max=6.0)
    hi_idx = torch.searchsorted(grid, a.contiguous()).clamp(max=len(grid) - 1)
    lo_idx = (hi_idx - 1).clamp(min=0)
    hi, lo = grid[hi_idx], grid[lo_idx]
    exact = hi == a
    span = torch.where(hi > lo, hi - lo, torch.ones_like(hi))
    frac = (a - lo) / span
    if stochastic:
        up = torch.rand_like(frac) < frac
    else:
        up = frac >= 0.5
    q = torch.where(exact, hi, torch.where(up, hi, lo)) * v.sign()
    return (q * s * g).reshape(shape).to(w.dtype)


def quant_modules(model):
    """Block linears and the YOCO cache; the embedding, value embeddings and head stay high precision."""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.startswith('transformer.') and not name.endswith('ve_gate'):
            out.append((name, m))
    return out


def install(model, mode):
    mods = quant_modules(model)
    if mode == 'master':
        for _, m in mods:
            m._qcache = (None, None)

            def fwd(self, x):
                key = (self.weight._version, self.weight.data_ptr())
                if self._qcache[0] != key:
                    self._qcache = (key, quantize(self.weight.detach()))
                return F.linear(x, self._qcache[1])
            m.forward = types.MethodType(fwd, m)
    elif mode == 'sr':
        requantize(model, stochastic=False)
    return mods


@torch.no_grad()
def requantize(model, stochastic=True):
    for _, m in quant_modules(model):
        m.weight.copy_(quantize(m.weight, stochastic=stochastic))


def quantize_rows_fp8(x):
    """Per-row (last dim) scaled e4m3 fake-quantization."""
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 448.0
    return (x / s).to(torch.float8_e4m3fn).float() * s


def quantize_delta(x, fmt):
    """Fake-quantize a perturbation tensor [..., d] with scales fitted to its own magnitude."""
    if fmt == 'fp8':
        return quantize_rows_fp8(x.float())
    flat = x.float().reshape(-1, x.shape[-1])
    return quantize(flat).reshape(x.shape)
