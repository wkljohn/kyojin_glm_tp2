"""Q8 linear check (one GPU, no model): hgemm_q8 vs the fp16 product of its dequantized weight, quantization error
vs the original fp16 weight, run-to-run bit equality, and speed against the fp16 wide GEMV on GLM decode shapes."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.fp16 import LinearFP16

torch.manual_seed(0)
dev = "cuda:0"
ok = True
for k, n in ((4096, 13056), (4352, 4096), (4096, 1024), (8704, 4096), (4096, 82176)):
    w = (torch.randn(k, n, device = dev) * 0.02).half()
    w[::97, ::13] *= 20                                     # outliers
    lin = LinearFP16(k, n, w.clone(), None, k, n, 0, 0)
    lin.quantize_q8()
    wd = lin._dequant()
    qerr = ((wd.float() - w.float()).norm() / w.float().norm()).item()
    times = {}
    for m in (1, 3, 8):
        x = torch.randn(m, k, device = dev).half()
        y = torch.empty(m, n, device = dev, dtype = torch.float)
        ext.hgemm_q8(x, lin.qweight, lin.qscale, y)
        y2 = torch.empty_like(y); ext.hgemm_q8(x, lin.qweight, lin.qscale, y2)
        ref = x.float() @ wd.float()
        rel = ((y - ref).norm() / ref.norm()).item()
        same = torch.equal(y, y2)
        ok &= rel < 1e-3 and same
        yf = torch.empty_like(y)
        def bench(f, it = 50):
            f(); torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(it): f()
            torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e6
        tq = bench(lambda: ext.hgemm_q8(x, lin.qweight, lin.qscale, y))
        tf = bench(lambda: ext.hgemm(x, w, yf))
        times[m] = (tq, tf, rel, same)
    print(f"k {k:5d} n {n:6d}  quant err {qerr:.2e}  " + "  ".join(
        f"M{m}: q8 {tq:6.0f} us vs fp16 {tf:6.0f} us (x{tf / tq:.2f}) err {rel:.1e} {'eq' if same else 'NONDET'}"
        for m, (tq, tf, rel, same) in times.items()), flush = True)
    # prefill path: > 8 rows dequantizes then runs the fp16 GEMM
    x = torch.randn(300, k, device = dev).half()
    y = lin.forward(x, {})
    ref = (x.float() @ wd.float())
    rel = ((y.float() - ref).norm() / ref.norm()).item()
    ok &= rel < 2e-3
    del lin, w, wd
    torch.cuda.empty_cache()
print("PASS" if ok else "FAIL")
