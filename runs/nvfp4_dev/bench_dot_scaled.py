"""Does Triton's MXFP4 tl.dot_scaled run natively (fast) on this GPU? Against an fp16 tl.dot of the same shape."""
import time
import torch
import triton
import triton.language as tl


@triton.jit
def k_fp16(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(a + rm[:, None] * K + (k0 + rk)[None, :])
        y = tl.load(b + (k0 + rk)[:, None] * N + rn[None, :])
        acc += tl.dot(x, y)
    tl.store(c + rm[:, None] * N + rn[None, :], acc)


@triton.jit
def k_fp4(a, sa, b, sb, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk2 = tl.arange(0, BK // 2)
    rs = tl.arange(0, BK // 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(a + rm[:, None] * (K // 2) + (k0 // 2 + rk2)[None, :])
        xs = tl.load(sa + rm[:, None] * (K // 32) + (k0 // 32 + rs)[None, :])
        y = tl.load(b + rn[:, None] * (K // 2) + (k0 // 2 + rk2)[None, :])
        ys = tl.load(sb + rn[:, None] * (K // 32) + (k0 // 32 + rs)[None, :])
        acc = tl.dot_scaled(x, xs, 'e2m1', y.T, ys, 'e2m1', acc)
    tl.store(c + rm[:, None] * N + rn[None, :], acc)


def t(f, n=20):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    s = time.time()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.time() - s) / n * 1e3


M, N, K = 65536, 768, 512
a = torch.randn(M, K, device='cuda', dtype=torch.half)
b = torch.randn(K, N, device='cuda', dtype=torch.half)
c = torch.empty(M, N, device='cuda')
a4 = torch.randint(0, 255, (M, K // 2), device='cuda', dtype=torch.uint8)
b4 = torch.randint(0, 255, (N, K // 2), device='cuda', dtype=torch.uint8)
sa = torch.full((M, K // 32), 127, device='cuda', dtype=torch.uint8)
sb = torch.full((N, K // 32), 127, device='cuda', dtype=torch.uint8)
g = (M // 128, N // 128)
print('fp16 tl.dot ms', t(lambda: k_fp16[g](a, b, c, M, N, K, BM=128, BN=128, BK=64)), flush=True)
try:
    print('mxfp4 tl.dot_scaled ms', t(lambda: k_fp4[g](a4, sa, b4, sb, c, M, N, K, BM=128, BN=128, BK=128)), flush=True)
except Exception as e:
    print('dot_scaled fail', str(e)[:300])
