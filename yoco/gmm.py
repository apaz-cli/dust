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

BM, BN, BK = 64, 256, 64  # tuned on the RTX 5090 at the expert shapes (65536 x 512 x 768 and back): about 187 TFLOP/s
NUM_WARPS, NUM_STAGES = 4, 2


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
                                   out.stride(0), out.stride(1), BM=BM, BN=BN, BK=BK,
                                   num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return out


# ---- MXFP4 grouped matmul (4-bit tensor cores through Triton's tl.dot_scaled; native on sm100 and sm120) -------
# MXFP4 (OCP microscaling): blocks of 32 values along K share a power-of-two scale (e8m0, stored as exponent + 127);
# each value is e2m1, two to a byte. Coarser scales than NVFP4 (16-value blocks, e4m3 scales), which Triton's scaled
# dot does not take; used here for the routed experts' delta forward, where only the change is quantized.
MX_BN, MX_BK = 128, 128


@triton.jit
def _mx_quant_kernel(x_ptr, q_ptr, s_ptr, M, K, BR: tl.constexpr, NB: tl.constexpr):
    """x [M, K] -> packed e2m1 [M, K/2] and e8m0 scales [M, K/32]; one program: BR rows x NB 32-wide blocks."""
    pr, pc = tl.program_id(0), tl.program_id(1)
    r = pr * BR + tl.arange(0, BR)
    k = pc * (NB * 32) + tl.arange(0, NB * 32)
    m = (r[:, None] < M) & (k[None, :] < K)
    v = tl.reshape(tl.load(x_ptr + r[:, None].to(tl.int64) * K + k[None, :], mask=m, other=0.0).to(tl.float32),
                   (BR, NB, 32))
    amax = tl.max(tl.abs(v), 2)
    # smallest power of two with amax / scale <= 6 (no saturation)
    e = tl.where(amax > 0, tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 6.0)), -127.0)
    e = tl.minimum(tl.maximum(e, -127.0), 127.0)
    scale = tl.exp2(e)
    a = tl.abs(v) / scale[:, :, None]
    code = (a > 0.25).to(tl.uint8) + (a > 0.75).to(tl.uint8) + (a > 1.25).to(tl.uint8) + (a > 1.75).to(tl.uint8) \
        + (a > 2.5).to(tl.uint8) + (a > 3.5).to(tl.uint8) + (a > 5.0).to(tl.uint8)
    code = code | (((v < 0) & (code > 0)).to(tl.uint8) << 3)
    lo, hi = tl.split(tl.reshape(code, (BR, NB * 16, 2)))
    kb = pc * (NB * 16) + tl.arange(0, NB * 16)
    tl.store(q_ptr + r[:, None].to(tl.int64) * (K // 2) + kb[None, :], lo | (hi << 4),
             mask=(r[:, None] < M) & (kb[None, :] < K // 2))
    c = pc * NB + tl.arange(0, NB)
    tl.store(s_ptr + r[:, None].to(tl.int64) * (K // 32) + c[None, :], (e + 127.0).to(tl.uint8),
             mask=(r[:, None] < M) & (c[None, :] < K // 32))


@triton_op("dust::mx_quant", mutates_args=())
def mx_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, K = x.shape
    q = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8)
    s = torch.empty(M, K // 32, device=x.device, dtype=torch.uint8)
    wrap_triton(_mx_quant_kernel)[(triton.cdiv(M, 64), triton.cdiv(K // 32, 4))](x, q, s, M, K, BR=64, NB=4)
    return q, s


def mx_dequant(q, s):
    """The values an MXFP4 matmul sees, fp32 (for checks)."""
    M, K2 = q.shape
    code = torch.stack([q & 0xF, q >> 4], -1).view(M, K2 * 2)
    grid = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=q.device)
    v = grid[(code & 7).long()] * torch.where(code >= 8, -1.0, 1.0)
    return (v.view(M, -1, 32) * torch.exp2(s.float() - 127).unsqueeze(-1)).view(M, K2 * 2)


@triton.jit
def _gmm_mx_kernel(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, tile_e, tile_r0, tile_r1, N, K,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """c[r] = a[r] @ b[e(r)].T in MXFP4: a packed [M, K/2] with scales [M, K/32]; b packed [E, N, K/2] with scales
    [E, N, K/32]; fp32 out [M, N]. Tiles never cross an expert boundary (tile table as in _gmm_kernel)."""
    t = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(tile_e + t)
    r0 = tl.load(tile_r0 + t)
    r1 = tl.load(tile_r1 + t)
    rm = r0 + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk2 = tl.arange(0, BK // 2)
    rs = tl.arange(0, BK // 32)
    row_ok = rm < r1
    col_ok = rn < N
    rm64 = rm.to(tl.int64)
    eb = e.to(tl.int64) * N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(a_ptr + rm64[:, None] * (K // 2) + (k0 // 2 + rk2)[None, :], mask=row_ok[:, None], other=0)
        xs = tl.load(as_ptr + rm64[:, None] * (K // 32) + (k0 // 32 + rs)[None, :], mask=row_ok[:, None], other=127)
        y = tl.load(b_ptr + (eb + rn)[:, None] * (K // 2) + (k0 // 2 + rk2)[None, :], mask=col_ok[:, None], other=0)
        ys = tl.load(bs_ptr + (eb + rn)[:, None] * (K // 32) + (k0 // 32 + rs)[None, :], mask=col_ok[:, None], other=127)
        acc = tl.dot_scaled(x, xs, 'e2m1', y.T, ys, 'e2m1', acc)
    tl.store(c_ptr + rm64[:, None] * N + rn[None, :], acc, mask=row_ok[:, None] & col_ok[None, :])


@triton_op("dust::gmm_mx", mutates_args=())
def gmm_mx(aq: torch.Tensor, a_s: torch.Tensor, bq: torch.Tensor, b_s: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """MXFP4 grouped matmul: aq [M, K/2] (rows sorted by expert, offs as for gmm), bq [E, N, K/2]; fp32 [M, N]."""
    M = aq.shape[0]
    K = aq.shape[1] * 2
    E, N, _ = bq.shape
    out = torch.empty(M, N, device=aq.device, dtype=torch.float32)
    te, r0, r1 = tiles(offs, M)
    wrap_triton(_gmm_mx_kernel)[(te.shape[0], triton.cdiv(N, MX_BN))](aq, a_s, bq, b_s, out, te, r0, r1, N, K,
                                                                      BM=BM, BN=MX_BN, BK=MX_BK, num_warps=4)
    return out


# ---- stable sort of the token slots by expert (a counting sort in two Triton kernels) ------------------------------
SORT_CHUNK = 1024


@triton.jit
def _expert_count_kernel(ids_ptr, cnt_ptr, S, E: tl.constexpr, CH: tl.constexpr):
    c = tl.program_id(0)
    i = c * CH + tl.arange(0, CH)
    e = tl.load(ids_ptr + i, mask=i < S, other=E)                       # E: padding, counted nowhere
    oh = (e[:, None] == tl.arange(0, E)[None, :]).to(tl.int32)         # [CH, E]
    tl.store(cnt_ptr + c * E + tl.arange(0, E), tl.sum(oh, 0))


@triton.jit
def _expert_place_kernel(ids_ptr, base_ptr, order_ptr, inv_ptr, S, E: tl.constexpr, CH: tl.constexpr):
    c = tl.program_id(0)
    i = c * CH + tl.arange(0, CH)
    ok = i < S
    e = tl.load(ids_ptr + i, mask=ok, other=0)
    oh = (e[:, None] == tl.arange(0, E)[None, :]).to(tl.int32)         # [CH, E]
    rank = tl.sum((tl.cumsum(oh, 0) - oh) * oh, 1)                     # rank among this chunk's slots of its expert
    base = tl.load(base_ptr + c * E + e, mask=ok, other=0)              # where this chunk's slots of that expert start
    pos = base + rank
    tl.store(inv_ptr + i, pos, mask=ok)
    tl.store(order_ptr + pos, i.to(tl.int64), mask=ok)


@triton_op("dust::sort_by_expert", mutates_args=())
def sort_by_expert(ids: torch.Tensor, E: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ids [S] (int64, values in 0..E-1, E a power of two) -> (order, inv, counts): the stable sort by expert,
    as argsort(stable=True) gives it, its inverse, and the slots per expert."""
    S = ids.shape[0]
    CH = max(64, SORT_CHUNK * 8 // max(E, 8))                           # [CH, E] must fit in shared memory
    nch = triton.cdiv(S, CH)
    cnt = torch.empty(nch, E, device=ids.device, dtype=torch.int32)
    wrap_triton(_expert_count_kernel)[(nch,)](ids, cnt, S, E=E, CH=CH)
    counts = cnt.sum(0).to(torch.int64)
    start = torch.cumsum(counts, 0) - counts                            # each expert's first position
    before = torch.cumsum(cnt, 0) - cnt                                 # that expert's slots in earlier chunks
    base = (start[None, :] + before).to(torch.int64).contiguous()
    order = torch.empty(S, device=ids.device, dtype=torch.int64)
    inv = torch.empty(S, device=ids.device, dtype=torch.int64)
    wrap_triton(_expert_place_kernel)[(nch,)](ids, base, order, inv, S, E=E, CH=CH)
    return order, inv, counts
