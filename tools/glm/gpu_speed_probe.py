"""Per-host GPU speed: fp16 GEMM TFLOPS and device memory bandwidth (sustained, 10 s each)."""
import time, socket, torch
a = torch.randn(8192, 8192, device = "cuda", dtype = torch.half)
def run(f, secs):
    f(); torch.cuda.synchronize(); n = 0; t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs:
        f(); n += 1
        if n % 4 == 0: torch.cuda.synchronize()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n
dt = run(lambda: torch.mm(a, a), 10)
x = torch.empty(512 * 2**20 // 2, device = "cuda", dtype = torch.half); y = torch.empty_like(x)
db = run(lambda: y.copy_(x), 10)
print(f"{socket.gethostname()}: gemm {2 * 8192**3 / dt / 1e12:.1f} TFLOPS  copy {2 * x.numel() * 2 / db / 1e9:.0f} GB/s", flush = True)
