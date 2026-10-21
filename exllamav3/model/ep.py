"""
Cross-host expert parallelism (EP) for one process per host.

Every rank runs the same model and the same inputs. Attention, KDA, dense MLPs, routers, mHC and the head are
replicated; each BlockSparseMLP keeps only its rank's contiguous expert range (permuted to local slots [0, n)
so the ROCm head-slice kernels mask the peer's picks to exact zeros) and the routed partials are summed with one
all-reduce per MoE layer. The shared expert runs on rank 0 only, so it is counted exactly once.

Enabled by environment (torchrun-style names are accepted too):
    KYOJIN_EP_WORLD=2 KYOJIN_EP_RANK=<r> MASTER_ADDR=<rank-0 address> MASTER_PORT=<port>
RCCL network selection is left to the usual NCCL_* variables (NCCL_NET=IB, NCCL_IB_HCA, NCCL_IB_GID_INDEX, ...).
"""
from __future__ import annotations
import os
import torch
import torch.distributed as dist

_STATE = None


class EPState:

    def __init__(self, rank: int, world: int):
        self.rank = rank
        self.world = world
        self.reduces = 0

    def expert_range(self, num_experts: int) -> tuple[int, int]:
        assert num_experts % self.world == 0, \
            f"EP needs the expert count ({num_experts}) divisible by the world size ({self.world})"
        n = num_experts // self.world
        return self.rank * n, (self.rank + 1) * n

    def expert_perm(self, num_experts: int) -> list[int]:
        """Router id order with this rank's experts first: perm[slot] = router expert id."""
        first, last = self.expert_range(num_experts)
        return list(range(first, last)) + [e for e in range(num_experts) if not first <= e < last]

    # Same signature as the TP backends' all_reduce (contribution=False: this rank adds zeros)
    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        if not contribution:
            tensor.zero_()
        if tensor.is_contiguous():
            dist.all_reduce(tensor)
        else:
            t = tensor.contiguous()
            dist.all_reduce(t)
            tensor.copy_(t)
        self.reduces += 1

    def broadcast_obj(self, obj = None, src: int = 0):
        box = [obj]
        dist.broadcast_object_list(box, src = src)
        return box[0]

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        dist.broadcast(tensor, src = src)

    def barrier(self):
        dist.barrier()


def _env_int(*names, default = None):
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return int(v)
    return default


def ep_state() -> EPState | None:
    """The process-wide EP state, initializing torch.distributed on first use; None when EP is off."""
    global _STATE
    if _STATE is not None:
        return _STATE or None
    world = _env_int("KYOJIN_EP_WORLD", default = 1)
    if world <= 1:
        _STATE = False
        return None
    rank = _env_int("KYOJIN_EP_RANK", "RANK")
    assert rank is not None and 0 <= rank < world, "KYOJIN_EP_RANK must be set when KYOJIN_EP_WORLD > 1"
    if not dist.is_initialized():
        addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = _env_int("MASTER_PORT", default = 29531)
        device = torch.device("cuda", torch.cuda.current_device())
        dist.init_process_group(
            "nccl",
            init_method = f"tcp://{addr}:{port}",
            rank = rank,
            world_size = world,
            device_id = device,
        )
        # First collective sets up the RCCL communicator; fail loudly here, not inside a layer
        t = torch.full((4096,), float(rank + 1), device = device)
        dist.all_reduce(t)
        expect = world * (world + 1) / 2
        assert float(t[0]) == expect, f"EP all-reduce self-test failed: {float(t[0])} != {expect}"
        torch.cuda.synchronize()
    _STATE = EPState(rank, world)
    print(f" -- EP rank {rank}/{world} ready (torch.distributed nccl)", flush = True)
    return _STATE
