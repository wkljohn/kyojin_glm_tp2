"""exl3_dec_router_rows (2..8 rows, one launch) vs per-row exl3_dec_router: bitwise selection and weights on the
GLM-5.3 router shape, and timing."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import dec_workspace

torch.manual_seed(0)
dev = "cuda:0"
H, E, K = 4096, 288, 8
G = (torch.randn(H, E, device = dev) * 0.02).half()
bias = (torch.randn(E, device = dev) * 0.01).float()
scratch, counters = dec_workspace(torch.device(dev))
ok = True
for trial in range(20):
    for R in range(2, 9):
        x = torch.randn(R, H, device = dev).half()
        if trial == 1:
            x[1] = x[0]                                     # identical rows
        s1 = torch.empty(R, K, dtype = torch.long, device = dev); w1 = torch.empty(R, K, dtype = torch.half, device = dev)
        for r in range(R):
            ext.exl3_dec_router(x[r:r + 1], G, bias, s1[r], w1[r], scratch, counters, 2.5)
        s2 = torch.empty_like(s1); w2 = torch.empty_like(w1)
        ext.exl3_dec_router_rows(x, G, bias, s2, w2, scratch, counters, 2.5)
        same = torch.equal(s1, s2) and torch.equal(w1.view(torch.int16), w2.view(torch.int16))
        ok &= same
        if not same:
            print(f"MISMATCH trial {trial} R {R}")
x = torch.randn(8, H, device = dev).half()
s = torch.empty(8, K, dtype = torch.long, device = dev); w = torch.empty(8, K, dtype = torch.half, device = dev)
def bench(f, it = 200):
    f(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e6
t8 = bench(lambda: ext.exl3_dec_router_rows(x, G, bias, s, w, scratch, counters, 2.5))
t44 = bench(lambda: (ext.exl3_dec_router_rows(x[:4], G, bias, s[:4], w[:4], scratch, counters, 2.5),
                     ext.exl3_dec_router_rows(x[4:], G, bias, s[4:], w[4:], scratch, counters, 2.5)))
t4 = bench(lambda: ext.exl3_dec_router_rows(x[:4], G, bias, s[:4], w[:4], scratch, counters, 2.5))
print(f"8 rows one launch {t8:.1f} us | 4+4 {t44:.1f} us | 4 rows {t4:.1f} us")
print("PASS" if ok else "FAIL")
