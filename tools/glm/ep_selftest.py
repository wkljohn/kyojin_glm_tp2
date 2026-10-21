"""EP transport self-test: RCCL all-reduce correctness and latency for decode / prefill payloads (fp32 rows of 4096)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import importlib.util
# Load ep.py standalone so the test runs without the compiled extension
_spec = importlib.util.spec_from_file_location("ep", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../exllamav3/model/ep.py"))
_ep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ep)
ep_state = _ep.ep_state

ep = ep_state()
assert ep is not None, "set KYOJIN_EP_WORLD=2"
for rows in (1, 4, 8, 64, 512, 2048):
    t = torch.full((rows, 4096), float(ep.rank + 1), device = "cuda")
    for _ in range(5): ep.all_reduce(t); t.fill_(ep.rank + 1)
    torch.cuda.synchronize()
    n = 200 if rows <= 64 else 20
    t0 = time.perf_counter()
    for _ in range(n):
        ep.all_reduce(t)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    t.fill_(ep.rank + 1); ep.all_reduce(t); torch.cuda.synchronize()
    ok = bool((t == 3.0).all())
    if ep.rank == 0:
        mb = rows * 4096 * 4 / 2**20
        print(f"rows {rows:5d} ({mb:7.3f} MiB): {dt * 1e3:8.3f} ms/all-reduce  {mb / 1024 / dt:6.2f} GiB/s  correct={ok}", flush = True)
torch.distributed.barrier()
torch.distributed.destroy_process_group()
