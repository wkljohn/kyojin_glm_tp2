"""Decode-size all-reduce latency through the pynccl communicator (KYOJIN_EP_PYNCCL=1): back-to-back (pipelined)
and synchronized (one op, then wait: the latency a layer sees) for bf16 rows of 4096. NCCL_* env tunes RCCL."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torch.distributed as dist
os.environ.setdefault("KYOJIN_EP_PYNCCL", "1")
from exllamav3.model.ep import ep_state
ep = ep_state()
pn = ep.pynccl


def bench(f, n, sync):
    for _ in range(20): f()
    torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
        if sync: torch.cuda.synchronize()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


for rows in (1, 2, 4, 8):
    x = torch.randn(rows, 4096, device = "cuda").bfloat16()
    a = bench(lambda: pn.all_reduce(x), 500, False)
    b = bench(lambda: pn.all_reduce(x), 500, True)
    if ep.rank == 0:
        print(f"rows {rows}: back-to-back {a:6.1f} us  synchronized {b:6.1f} us", flush = True)
dist.barrier()
