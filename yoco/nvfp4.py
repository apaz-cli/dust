"""Real NVFP4 matrix multiplies (Blackwell tensor cores: RTX 5090 sm120, B300 sm100), for Dust's forward reruns.

NVFP4 (NVIDIA): blocks of 16 consecutive values along the contraction dimension share an FP8 (e4m3) scale; each value
is FP4 e2m1, one of {0, 0.5, 1, 1.5, 2, 3, 4, 6} with a sign, two to a byte; a per-tensor fp32 scale keeps the block
scales inside e4m3's range. Round to nearest, as in the forward pass of the published NVFP4 training recipes
(Xue et al. 2025, arXiv 2509.25149; Chmiel et al. 2025, arXiv 2505.19115).

`quantize(x)` -> QTensor (packed values, block scales in the swizzled layout cuBLAS wants, per-tensor scale) for a 2-D
tensor [rows, K] quantized along K. `mm(a, b)` multiplies a [M, K] by b [N, K] transposed on the FP4 tensor cores
(torch._scaled_mm) and returns fp32 [M, N]. `dequantize(q)` gives back the values the matmul sees, for testing.
"""
from dataclasses import dataclass

import torch

BLOCK = 16
E4M3_MAX = 448.0
E2M1_MAX = 6.0
# round to nearest on the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}: midpoints between neighbours
_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass
class QTensor:
    data: torch.Tensor    # [rows, K // 2], torch.float4_e2m1fn_x2 (two values per byte, the first in the low nibble)
    scale: torch.Tensor   # block scales, e4m3, swizzled (cuBLAS 128x4 block layout), flattened
    raw_scale: torch.Tensor  # the same block scales [rows, K // 16], plain layout (for dequantize)
    gscale: torch.Tensor  # per-tensor fp32 scale (0-dim)
    shape: tuple


def _swizzle(s):
    """Block scales [rows, K/16] -> the 128x4-tiled layout cuBLAS expects for block-scaled FP4 (as torchao's to_blocked)."""
    rows, cols = s.shape
    R, C = -(-rows // 128), -(-cols // 4)
    if (R * 128, C * 4) != (rows, cols):
        s = torch.nn.functional.pad(s, (0, C * 4 - cols, 0, R * 128 - rows))
    blocks = s.view(R, 128, C, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def quantize(x):
    """x: [rows, K] (K a multiple of 16), any float dtype -> QTensor."""
    rows, K = x.shape
    xf = x.float()
    g = (xf.abs().amax() / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-30)
    xb = xf.view(rows, K // BLOCK, BLOCK)
    s = (xb.abs().amax(-1) / (E2M1_MAX * g)).to(torch.float8_e4m3fn)        # block scale, in units of g
    sf = s.float().clamp(min=1e-30)
    v = xb / (sf * g).unsqueeze(-1)                                          # in [-6, 6]
    bounds = torch.tensor(_BOUNDS, device=x.device)
    code = torch.bucketize(v.abs(), bounds).to(torch.uint8)                  # 0..7, nearest grid point
    code = code | ((v < 0) & (code > 0)).to(torch.uint8) << 3                # sign bit
    code = code.view(rows, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)
    return QTensor(packed, _swizzle(s), s, g, (rows, K))


def dequantize(q):
    """The values the FP4 matmul sees, as fp32 [rows, K] (for checks)."""
    rows, K = q.shape
    b = q.data.view(torch.uint8)
    code = torch.stack([b & 0xF, b >> 4], -1).view(rows, K)
    grid = torch.tensor(_GRID, device=b.device)
    v = grid[(code & 7).long()] * torch.where(code >= 8, -1.0, 1.0)
    return (v.view(rows, K // BLOCK, BLOCK) * q.raw_scale.float().unsqueeze(-1) * q.gscale).view(rows, K)


def mm(a, b, out_dtype=torch.float32):
    """a: QTensor [M, K], b: QTensor [N, K] -> a @ b.T, [M, N], on the FP4 tensor cores."""
    out = torch._scaled_mm(a.data, b.data.t(), a.scale, b.scale, out_dtype=torch.bfloat16)
    return (out.float() * (a.gscale * b.gscale)).to(out_dtype)


# ---- fast fused quantization (Triton): optional subtraction of a clean tensor, scales written pre-swizzled --------
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _amax_kernel(x_ptr, base_ptr, out_ptr, M, K, base_rows, HAS_BASE: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr):
    pr, pk = tl.program_id(0), tl.program_id(1)
    r = pr * BR + tl.arange(0, BR)
    k = pk * BK + tl.arange(0, BK)
    m = (r[:, None] < M) & (k[None, :] < K)
    v = tl.load(x_ptr + r[:, None].to(tl.int64) * K + k[None, :], mask=m, other=0.0).to(tl.float32)
    if HAS_BASE:
        br = r % base_rows
        v -= tl.load(base_ptr + br[:, None].to(tl.int64) * K + k[None, :], mask=m, other=0.0).to(tl.float32)
    tl.atomic_max(out_ptr, tl.max(tl.max(tl.abs(v), 1), 0))


@triton.jit
def _quant_kernel(x_ptr, base_ptr, g_ptr, q_ptr, s_ptr, M, K, base_rows, n_scale_cols_tiles,
                  HAS_BASE: tl.constexpr, BR: tl.constexpr):
    """One program: BR rows x one 16-wide block column."""
    pr, pc = tl.program_id(0), tl.program_id(1)
    r = pr * BR + tl.arange(0, BR)
    k = pc * 16 + tl.arange(0, 16)
    m = r[:, None] < M
    v = tl.load(x_ptr + r[:, None].to(tl.int64) * K + k[None, :], mask=m, other=0.0).to(tl.float32)
    if HAS_BASE:
        br = r % base_rows
        v -= tl.load(base_ptr + br[:, None].to(tl.int64) * K + k[None, :], mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr)
    amax = tl.max(tl.abs(v), 1)                                    # [BR]
    s = (amax / (6.0 * g)).to(tl.float8e4nv)                       # block scale, e4m3
    sf = tl.maximum(s.to(tl.float32), 1e-30) * g
    a = tl.abs(v) / sf[:, None]
    code = (a > 0.25).to(tl.uint8) + (a > 0.75).to(tl.uint8) + (a > 1.25).to(tl.uint8) + (a > 1.75).to(tl.uint8) \
        + (a > 2.5).to(tl.uint8) + (a > 3.5).to(tl.uint8) + (a > 5.0).to(tl.uint8)
    code = code | (((v < 0) & (code > 0)).to(tl.uint8) << 3)
    lo, hi = tl.split(tl.reshape(code, (BR, 8, 2)))
    byte = lo | (hi << 4)                                          # [BR, 8]
    kb = pc * 8 + tl.arange(0, 8)
    tl.store(q_ptr + r[:, None].to(tl.int64) * (K // 2) + kb[None, :], byte, mask=m)
    # swizzled scale offset (cuBLAS 128x4 tiles): tile (r // 128, c // 4), inside it (r % 32) * 16 + (r % 128 // 32) * 4 + c % 4
    tile = (r // 128) * n_scale_cols_tiles + pc // 4
    off = tile.to(tl.int64) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + pc % 4
    tl.store(s_ptr + off, s, mask=r < M)


def quantize_fast(x, base=None):
    """x [M, K] (fp16/bf16/fp32, contiguous), minus base [M', K] repeated over rows (M a multiple of M') if given ->
    QTensor (raw_scale not filled: use quantize() for checks). Two passes: the tensor's max, then the blocks."""
    x = x.contiguous()
    M, K = x.shape
    has_base = base is not None
    bptr = base.contiguous() if has_base else x
    brows = bptr.shape[0]
    amax = torch.zeros((), device=x.device, dtype=torch.float32)
    _amax_kernel[(triton.cdiv(M, 64), triton.cdiv(K, 128))](x, bptr, amax, M, K, brows, HAS_BASE=has_base, BR=64, BK=128)
    g = (amax / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-30)
    q = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8)
    R, C = triton.cdiv(M, 128), triton.cdiv(K // BLOCK, 4)
    s = torch.zeros(R * C * 512, device=x.device, dtype=torch.float8_e4m3fn)
    _quant_kernel[(triton.cdiv(M, 64), K // BLOCK)](x, bptr, g, q, s, M, K, brows, C, HAS_BASE=has_base, BR=64)
    return QTensor(q.view(torch.float4_e2m1fn_x2), s, None, g, (M, K))


# ---- custom ops so torch.compile can trace the quantizer (the reruns run compiled) -------------------------------
from torch.library import triton_op, wrap_triton  # noqa: E402


@triton_op("dust::fp4_quant_delta", mutates_args=())
def fp4_quant_delta(x: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x - base repeated over rows) -> (packed fp4 as uint8 [M, K/2], swizzled e4m3 scales, per-tensor fp32 scale)."""
    M, K = x.shape
    brows = base.shape[0]
    amax = torch.zeros((1,), device=x.device, dtype=torch.float32)
    wrap_triton(_amax_kernel)[(triton.cdiv(M, 64), triton.cdiv(K, 128))](x, base, amax, M, K, brows,
                                                                          HAS_BASE=True, BR=64, BK=128)
    g = (amax / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-30)
    q = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8)
    C = triton.cdiv(K // BLOCK, 4)
    s = torch.zeros(triton.cdiv(M, 128) * C * 512, device=x.device, dtype=torch.float8_e4m3fn)
    wrap_triton(_quant_kernel)[(triton.cdiv(M, 64), K // BLOCK)](x, base, g, q, s, M, K, brows, C,
                                                                 HAS_BASE=True, BR=64)
    return q, s, g


def delta_linear(x, x_clean, y_clean, wq):
    """y_clean (repeated over draws) + FP4(x - x_clean) @ FP4(W).T. x: [..., K] with rows n * rows(x_clean);
    wq: QTensor of the weight. Returns fp32 [..., N]."""
    K = x.shape[-1]
    X = x.reshape(-1, K).contiguous()
    q, s, g = torch.ops.dust.fp4_quant_delta(X, x_clean)
    d = torch._scaled_mm(q.view(torch.float4_e2m1fn_x2), wq.data.t(), s, wq.scale, out_dtype=torch.bfloat16)
    n = X.shape[0] // y_clean.shape[0]
    out = y_clean.float().repeat(n, 1) + d.float() * (g * wq.gscale)
    return out.view(*x.shape[:-1], -1)


@triton_op("dust::fp4_quant", mutates_args=())
def fp4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, K] -> (packed fp4 as uint8, swizzled e4m3 scales, per-tensor fp32 scale)."""
    M, K = x.shape
    amax = torch.zeros((1,), device=x.device, dtype=torch.float32)
    wrap_triton(_amax_kernel)[(triton.cdiv(M, 64), triton.cdiv(K, 128))](x, x, amax, M, K, M,
                                                                          HAS_BASE=False, BR=64, BK=128)
    g = (amax / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-30)
    q = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8)
    C = triton.cdiv(K // BLOCK, 4)
    s = torch.zeros(triton.cdiv(M, 128) * C * 512, device=x.device, dtype=torch.float8_e4m3fn)
    wrap_triton(_quant_kernel)[(triton.cdiv(M, 64), K // BLOCK)](x, x, g, q, s, M, K, M, C, HAS_BASE=False, BR=64)
    return q, s, g


def delta_linear_w(x, x_clean, y_clean, w):
    """delta_linear with the weight quantized here (cheap next to the activations: a weight is [N, K])."""
    K = x.shape[-1]
    X = x.reshape(-1, K).contiguous()
    q, s, g = torch.ops.dust.fp4_quant_delta(X, x_clean)
    wq, ws, wg = torch.ops.dust.fp4_quant(w.contiguous())
    d = torch._scaled_mm(q.view(torch.float4_e2m1fn_x2), wq.view(torch.float4_e2m1fn_x2).t(), s, ws,
                         out_dtype=torch.bfloat16)
    n = X.shape[0] // y_clean.shape[0]
    out = y_clean.float().repeat(n, 1) + d.float() * (g * wg)
    return out.view(*x.shape[:-1], -1)


# ---- v2: one pass, larger tiles, the delta's per-tensor scale fixed from the clean input --------------------------
@triton.jit
def _quant2_kernel(x_ptr, base_ptr, g_ptr, q_ptr, s_ptr, M, K, base_rows, n_scale_cols_tiles,
                   HAS_BASE: tl.constexpr, BR: tl.constexpr, NB: tl.constexpr):
    """One program: BR rows x NB 16-wide blocks."""
    pr, pc = tl.program_id(0), tl.program_id(1)
    r = pr * BR + tl.arange(0, BR)
    k = pc * (NB * 16) + tl.arange(0, NB * 16)
    m = (r[:, None] < M) & (k[None, :] < K)
    v = tl.load(x_ptr + r[:, None].to(tl.int64) * K + k[None, :], mask=m, other=0.0).to(tl.float32)
    if HAS_BASE:
        br = r % base_rows
        v -= tl.load(base_ptr + br[:, None].to(tl.int64) * K + k[None, :], mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr)
    v3 = tl.reshape(v, (BR, NB, 16))
    amax = tl.max(tl.abs(v3), 2)                                        # [BR, NB]
    s = tl.minimum(amax / (6.0 * g), 448.0).to(tl.float8e4nv)           # clamp: the scale is fixed in advance
    sf = tl.maximum(s.to(tl.float32), 1e-30) * g
    a = tl.abs(v3) / sf[:, :, None]
    code = (a > 0.25).to(tl.uint8) + (a > 0.75).to(tl.uint8) + (a > 1.25).to(tl.uint8) + (a > 1.75).to(tl.uint8) \
        + (a > 2.5).to(tl.uint8) + (a > 3.5).to(tl.uint8) + (a > 5.0).to(tl.uint8)
    code = code | (((v3 < 0) & (code > 0)).to(tl.uint8) << 3)
    lo, hi = tl.split(tl.reshape(code, (BR, NB * 8, 2)))
    byte = lo | (hi << 4)                                               # [BR, NB * 8]
    kb = pc * (NB * 8) + tl.arange(0, NB * 8)
    tl.store(q_ptr + r[:, None].to(tl.int64) * (K // 2) + kb[None, :], byte, mask=(r[:, None] < M) & (kb[None, :] < K // 2))
    c = pc * NB + tl.arange(0, NB)                                      # block-column index
    tile = (r[:, None] // 128) * n_scale_cols_tiles + c[None, :] // 4
    off = tile.to(tl.int64) * 512 + (r[:, None] % 32) * 16 + ((r[:, None] % 128) // 32) * 4 + c[None, :] % 4
    tl.store(s_ptr + off, s, mask=(r[:, None] < M) & (c[None, :] < K // 16))


@triton_op("dust::fp4_quant_delta2", mutates_args=())
def fp4_quant_delta2(x: torch.Tensor, base: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(x - base repeated over rows), per-tensor scale g given -> (packed fp4 as uint8 [M, K/2], swizzled scales)."""
    M, K = x.shape
    q = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8)
    C = triton.cdiv(K // BLOCK, 4)
    s = torch.zeros(triton.cdiv(M, 128) * C * 512, device=x.device, dtype=torch.float8_e4m3fn)
    NB = 8
    wrap_triton(_quant2_kernel)[(triton.cdiv(M, 64), triton.cdiv(K // BLOCK, NB))](
        x, base, g, q, s, M, K, base.shape[0], C, HAS_BASE=True, BR=64, NB=NB)
    return q, s


def delta_scale(x_clean):
    """Per-tensor scale for the deltas of a rerun: from the clean input's largest value (a perturbation's change is
    smaller than the activation; block scales above e4m3's range are clamped)."""
    return (2.0 * x_clean.abs().amax().float() / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-30).reshape(1)


def delta_linear_q(x, x_clean, y_clean, g, w_data, w_scale, w_g):
    """delta_linear with a pre-quantized weight (its packed values, swizzled scales and per-tensor scale) and a fixed
    per-tensor scale g for the delta."""
    K = x.shape[-1]
    X = x.reshape(-1, K).contiguous()
    q, s = torch.ops.dust.fp4_quant_delta2(X, x_clean, g)
    d = torch._scaled_mm(q.view(torch.float4_e2m1fn_x2), w_data.t(), s, w_scale, out_dtype=torch.bfloat16)
    n = X.shape[0] // y_clean.shape[0]
    out = y_clean.float().repeat(n, 1) + d.float() * (g * w_g)
    return out.view(*x.shape[:-1], -1)


def fake_quant_rows(x):
    """NVFP4 round trip in plain torch (traceable): x [rows, K] -> the fp32 values an FP4 matmul would see, quantized in
    1 x 16 blocks along K with a per-tensor scale."""
    rows, K = x.shape
    xf = x.float()
    g = (xf.abs().amax() / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-30)
    xb = xf.view(rows, K // BLOCK, BLOCK)
    s = (xb.abs().amax(-1, keepdim=True) / (E2M1_MAX * g)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).float()
    sf = s.clamp(min=1e-30) * g
    a = (xb / sf).abs()
    bounds = torch.tensor(_BOUNDS, device=x.device)
    grid = torch.tensor(_GRID, device=x.device)
    v = grid[torch.bucketize(a, bounds)] * torch.sign(xb)
    return (v * sf).view(rows, K)
