"""Q8 GEMV in the DRAM-bound regime: rotate over enough weight copies (> 512 MB) that no call hits the last-level
cache, as in a real decode round. Reports GB/s at 1 and 8 rows per GLM decode shape."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.fp16 import LinearFP16

dev = "cuda:0"
shapes = [(4096, 13056), (4352, 4096), (4096, 1024), (8704, 4096), (4096, 16384), (16384, 4096), (4096, 6144)]
shapes = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1:]] or shapes
for k, n in shapes:
    w = (torch.randn(k, n, device = dev) * 0.02).half()
    base = LinearFP16(k, n, w, None, k, n, 0, 0)
    base.quantize_q8()
    nbytes = base.qweight.numel() + base.qscale.numel() * 2
    copies = max(2, (512 << 20) // nbytes)
    qs = [(base.qweight.clone(), base.qscale.clone()) for _ in range(copies)]
    del w
    res = []
    for m in (1, 8):
        x = torch.zeros(m, k, device = dev).half() if os.environ.get("Q8B_ZEROX") else torch.randn(m, k, device = dev).half()
        y = torch.empty(m, n, device = dev, dtype = torch.float)
        for q, s in qs: ext.hgemm_q8(x, q, s, y)
        torch.cuda.synchronize(); t0 = time.perf_counter(); it = 0
        for _ in range(3):
            for q, s in qs: ext.hgemm_q8(x, q, s, y); it += 1
        torch.cuda.synchronize(); t = (time.perf_counter() - t0) / it
        res.append((m, t * 1e6, nbytes / t / 1e9))
    print(f"k {k:5d} n {n:6d} {nbytes / 2**20:6.1f} MiB  " + "  ".join(f"M{m}: {us:7.1f} us {gb:5.0f} GB/s" for m, us, gb in res)
          + f"  M8/M1 {res[1][1] / res[0][1]:.2f}", flush = True)
    del qs, base
    torch.cuda.empty_cache()
