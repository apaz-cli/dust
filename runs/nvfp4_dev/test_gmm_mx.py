import sys, time, torch
sys.path.insert(0, '/home/apaz/git/dust_experiments')
from yoco import gmm as G
torch.manual_seed(0)
E, K, N, M = 8, 512, 768, 20000
ids = torch.randint(0, E, (M,), device='cuda'); ids[ids == 3] = 4
ids, _ = ids.sort(); offs = torch.bincount(ids, minlength=E).cumsum(0).int()
a = torch.randn(M, K, device='cuda') * torch.rand(M, 1, device='cuda')
w = torch.randn(E, N, K, device='cuda') * 0.05                          # [E, N, K]: out x in
aq, as_ = G.mx_quant(a)
wq, ws = G.mx_quant(w.reshape(E * N, K)); wq, ws = wq.view(E, N, K // 2), ws.view(E, N, K // 32)
got = G.gmm_mx(aq, as_, wq, ws, offs)
ad = G.mx_dequant(aq, as_); wd = G.mx_dequant(wq.reshape(E * N, -1), ws.reshape(E * N, -1)).view(E, N, K)
ref = torch.cat([ad[ids == e] @ wd[e].t() for e in range(E)])
exact = torch.cat([a[ids == e] @ w[e].t() for e in range(E)])
print('kernel vs dequant ref rel err', ((got - ref).norm() / ref.norm()).item(), ' mxfp4 vs exact', ((got - exact).norm() / exact.norm()).item())
M2 = 65536
ids = torch.randint(0, E, (M2,), device='cuda').sort().values; offs = torch.bincount(ids, minlength=E).cumsum(0).int()
a = torch.randn(M2, K, device='cuda', dtype=torch.half); aq, as_ = G.mx_quant(a)
b16 = torch.randn(E, K, N, device='cuda', dtype=torch.half)
def t(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); s = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - s) / n * 1e3
print('fp16 gmm ms', t(lambda: G.gmm(a, b16, offs)), ' mxfp4 gmm ms', t(lambda: G.gmm_mx(aq, as_, wq, ws, offs)), ' mx quant ms', t(lambda: G.mx_quant(a)))
