"""Can an RCCL all-reduce (16 MiB bf16, the 2048-row prefill payload) run while the compute stream works?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import importlib.util
import torch, torch.distributed as dist
_spec = importlib.util.spec_from_file_location("ep", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../exllamav3/model/ep.py"))
_ep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ep)
ep = _ep.ep_state()
a = torch.randn(4096, 4096, device = "cuda", dtype = torch.half)
c = torch.randn(2048, 4096, device = "cuda", dtype = torch.bfloat16)
def compute(k):
    for _ in range(k): torch.mm(a, a)
def timed(f, n = 5):
    f(); torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3
K = 8
tc = timed(lambda: compute(K))
tr = timed(lambda: [dist.all_reduce(c) for _ in range(4)])
def both():
    ws = [dist.all_reduce(c, async_op = True) for _ in range(4)]
    compute(K)
    for w in ws: w.wait()
tb = timed(both)
if ep.rank == 0:
    print(f"compute {tc:.1f} ms  4x all-reduce {tr:.1f} ms  both {tb:.1f} ms  (serial {tc + tr:.1f}, ideal {max(tc, tr):.1f})", flush = True)
dist.barrier(); dist.destroy_process_group()
