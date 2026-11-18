"""Grouped matmul for the routed experts, as a Triton kernel that torch.compile can trace (torch.library.triton_op).

out[r] = a[r] @ b[e(r)] for the rows r of a, which are sorted by expert: expert e owns rows offs[e-1] .. offs[e]-1.
a [M, K] (fp16 or bf16), b [E, K, N] (same dtype, any strides), offs [E] int32 (cumulative row counts). Returns fp32
[M, N], accumulated in fp32. Rows are cut into tiles of BM rows that never cross an expert boundary; the tile table is
built on the device with fixed sizes (at most ceil(M / BM) + E tiles), so there is no host sync and nothing
data-dependent in any shape. Forward only (Dust needs no gradients).

Why not torch._grouped_mm: it takes bf16 only, and in the torch 2.10 wheels its sm100 (B300) kernel fails to launch
("TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED").
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

BM, BN, BK = 64, 64, 32


@triton.jit
def _gmm_kernel(a_ptr, b_ptr, c_ptr, tile_e, tile_r0, tile_r1, N, K,
                sam, sak, sbe, sbk, sbn, scm, scn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(tile_e + t)
    r0 = tl.load(tile_r0 + t)
    r1 = tl.load(tile_r1 + t)  # r1 == r0 for an unused tile: everything masked
    rm = r0 + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    row_ok = rm < r1
    col_ok = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(a_ptr + rm[:, None].to(tl.int64) * sam + kk[None, :] * sak,
                    mask=row_ok[:, None] & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + e.to(tl.int64) * sbe + kk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(kk[:, None] < K) & col_ok[None, :], other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None].to(tl.int64) * scm + rn[None, :] * scn, acc, mask=row_ok[:, None] & col_ok[None, :])


def tiles(offs, M):
    """Tile table: for each of the ceil(M / BM) + E tiles its expert, first row and end row (fixed sizes)."""
    E = offs.shape[0]
    ends = offs.to(torch.int64)
    counts = ends - torch.cat([ends.new_zeros(1), ends[:-1]])
    per = (counts + BM - 1) // BM
    cum = torch.cumsum(per, 0)
    T = (M + BM - 1) // BM + E
    t = torch.arange(T, device=offs.device)
    e = torch.searchsorted(cum, t, right=True)
    valid = e < E
    ec = e.clamp(max=E - 1)
    r0 = (ends - counts)[ec] + (t - (cum - per)[ec]) * BM
    r1 = torch.minimum(r0 + BM, ends[ec])
    r1 = torch.where(valid, r1, r0)
    return ec.to(torch.int32), r0.to(torch.int32), r1.to(torch.int32)


@triton_op("dust::gmm", mutates_args=())
def gmm(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    E, _, N = b.shape
    out = torch.empty(M, N, device=a.device, dtype=torch.float32)
    te, r0, r1 = tiles(offs, M)
    grid = (te.shape[0], triton.cdiv(N, BN))
    wrap_triton(_gmm_kernel)[grid](a, b, out, te, r0, r1, N, K,
                                   a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2),
                                   out.stride(0), out.stride(1), BM=BM, BN=BN, BK=BK)
    return out
