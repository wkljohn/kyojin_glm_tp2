"""Per-rank device memory after loading the target (+ drafter) the way the EP tools do, before any cache.
    tools/glm/ep_run2.sh <log> tools/glm/ep_mem.py <model dir> <draft dir>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ep_bench  # noqa: F401  (applies the GLM SPEED_ENV)
import torch
from exllamav3 import Config, Model
from exllamav3.model.ep import ep_state

ep = ep_state()


def mem(tag):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"[mem] rank {ep.rank} {tag}: used {(total - free) / 2**30:.2f} GiB, torch allocated "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} reserved {torch.cuda.memory_reserved() / 2**30:.2f}", flush = True)


mem("start")
m = Model.from_config(Config.from_directory(sys.argv[1]))
m.load(device = "cuda:0", progressbar = False)
mem("target")
d = Model.from_config(Config.from_directory(sys.argv[2]))
d.load(device = "cuda:0", progressbar = False)
mem("target+draft")
torch.distributed.barrier()
