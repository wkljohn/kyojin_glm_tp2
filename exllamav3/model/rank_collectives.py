"""Collectives for one independently loaded model rank on each of two hosts.

The caller owns process-group initialization and transport selection. This is
the broadcast/reduction interface used by rank-local GLM MoE modules; model
loading and session commands do not use Kyojin's shared-memory TP launcher.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


class RankCollectives:
    def __init__(self, group=None):
        if not dist.is_initialized():
            raise RuntimeError("Initialize the distributed process group first")
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.world_size != 2 or str(dist.get_backend(group)) != "nccl":
            raise ValueError("Rank-local GLM currently requires two NCCL/RCCL ranks")

    @staticmethod
    def _contiguous(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cuda":
            raise ValueError("Rank collectives require GPU tensors")
        return tensor if tensor.is_contiguous() else tensor.contiguous()

    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True) -> None:
        """Sum in place, retaining the tensor's dtype and zeroing noncontributors."""
        value = self._contiguous(tensor)
        if not contribution:
            value.zero_()
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=self.group)
        if value is not tensor:
            tensor.copy_(value)

    def broadcast(self, tensor: torch.Tensor, src_device: int = 0) -> None:
        """Broadcast from a global rank; the legacy keyword is ``src_device``.

        Each host can use local GPU zero. Device ordinals must never determine
        which process owns a contribution or a routing decision.
        """
        if not 0 <= src_device < self.world_size:
            raise ValueError("Broadcast source is outside the rank group")
        value = self._contiguous(tensor)
        dist.broadcast(value, group=self.group, group_src=src_device)
        if value is not tensor:
            tensor.copy_(value)
