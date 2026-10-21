"""Can an RCCL all-reduce be captured in a HIP graph with compute around it (torch.cuda.graph), and replayed?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import importlib.util
import torch, torch.distributed as dist
_spec = importlib.util.spec_from_file_location("ep", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../exllamav3/model/ep.py"))
_ep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ep)
ep = _ep.ep_state()
w = torch.randn(4096, 4096, device = "cuda", dtype = torch.half) * 0.01
x = torch.randn(8, 4096, device = "cuda", dtype = torch.half)
def step():
    y = x
    for _ in range(45):
        y = (y @ w).float()
        dist.all_reduce(y)
        y = (y * 0.5).half()
    return y
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): step()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
ref = step(); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        out = step()
    g.replay(); torch.cuda.synchronize()
    ok = torch.equal(out, ref)
    def bench(f, n = 50):
        f(); torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
        for _ in range(n): f()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
    te, tg = bench(step), bench(g.replay)
    # interleave graph replays with eager collectives on the same communicator (the block-graph pattern)
    z = torch.ones(8, 82048, device = "cuda")
    zz = torch.empty(2, 8, 82048, device = "cuda")
    for i in range(200):
        g.replay()
        dist.all_reduce(z[:1, :4096])
        dist.all_gather_into_tensor(zz.view(-1), z.view(-1))
        if i % 50 == 0:
            torch.cuda.synchronize()
            print(f"rank {ep.rank}: interleave {i} ok", flush = True)
    torch.cuda.synchronize()
    print(f"rank {ep.rank}: capture OK, replay equal={ok}; eager {te:.2f} ms vs graph {tg:.2f} ms per 45-layer step", flush = True)
except Exception as e:
    print(f"rank {ep.rank}: capture FAILED: {type(e).__name__}: {str(e)[:300]}", flush = True)
dist.barrier(); dist.destroy_process_group()
