"""Q8 vs fp16 wide GEMV with cold weights (8 rotating copies > the 32 MB MALL), GLM decode shapes."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.fp16 import LinearFP16
dev = "cuda:0"; C = 8
for k, n in ((4352, 4096), (4096, 13056), (4096, 1024), (4096, 6656), (8704, 4096)):
    lins, ws = [], []
    for _ in range(C):
        w = (torch.randn(k, n, device = dev) * 0.02).half()
        l = LinearFP16(k, n, w.clone(), None, k, n, 0, 0); l.quantize_q8(); lins.append(l); ws.append(w)
    out = []
    for m in (1, 8):
        x = torch.randn(m, k, device = dev).half(); y = torch.empty(m, n, device = dev)
        def bench(f):
            for i in range(C): f(i)
            torch.cuda.synchronize(); t0 = time.perf_counter(); it = 40
            for j in range(it): f(j % C)
            torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e6
        tq = bench(lambda i: ext.hgemm_q8(x, lins[i].qweight, lins[i].qscale, y))
        tf = bench(lambda i: ext.hgemm(x, ws[i], y))
        out.append(f"M{m}: q8 {tq:5.0f} us {k * n / tq / 1e3:4.0f} GB/s | fp16 {tf:5.0f} us {2 * k * n / tf / 1e3:4.0f} GB/s")
    print(f"blocks {os.environ.get('EXL3_Q8_BLOCKS', 'def')} k {k} n {n}: " + "   ".join(out), flush = True)
    del lins, ws; torch.cuda.empty_cache()
