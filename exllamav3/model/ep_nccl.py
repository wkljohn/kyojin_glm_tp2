"""
Thin ctypes RCCL communicator for the EP decode all-reduces (KYOJIN_EP_PYNCCL=1), after vLLM's PyNcclCommunicator.

torch's ProcessGroupNCCL spends ~80 us of host time per all_reduce call (work objects, events, watchdog
bookkeeping). A decode round issues ~90 of them, and the host is close to the critical path there, so the per-call
cost shows up as GPU idle time. ncclAllReduce on the current stream costs a few us. The communicator is a second one
beside torch's: both ranks issue every collective on the same stream in the same order, so mixing them is safe.
"""
from __future__ import annotations
import ctypes
import os
import torch
import torch.distributed as dist


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


_DTYPES = {torch.float32: 7, torch.float16: 6, torch.bfloat16: 9, torch.int64: 4, torch.int32: 2, torch.float64: 8}
_OPS = {"sum": 0, "max": 2, "min": 3}


def _find_lib() -> str:
    path = os.environ.get("KYOJIN_RCCL_LIB")
    if path:
        return path
    # The librccl torch itself loaded (same process, same build)
    for line in open("/proc/self/maps"):
        if "librccl.so" in line:
            return line.split()[-1]
    return "librccl.so.1"


class PyNccl:

    def __init__(self, rank: int, world: int, device: torch.device):
        self.lib = lib = ctypes.CDLL(_find_lib())
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int]
        lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_void_p]
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        self.device = device
        self.comm = self._init_comm(rank, world, device)
        # KYOJIN_EP_PYNCCL_SIMPLE_MIN (bytes, default 0 = off): payloads from this size go through a second
        # communicator created with NCCL_PROTO=Simple (Simple beats LL from 2 decode rows on: 4 rows bf16 44.5 vs 54 us).
        # Measured no effect: this RCCL reads NCCL_PROTO once per process, so the second comm keeps the first's choice
        self.simple_min = int(os.environ.get("KYOJIN_EP_PYNCCL_SIMPLE_MIN", 0))
        self.comm_simple = None
        if self.simple_min:
            old = os.environ.get("NCCL_PROTO")
            os.environ["NCCL_PROTO"] = "Simple"
            try:
                self.comm_simple = self._init_comm(rank, world, device)
            finally:
                if old is None: del os.environ["NCCL_PROTO"]
                else: os.environ["NCCL_PROTO"] = old

    def _init_comm(self, rank, world, device):
        lib = self.lib
        uid = _UniqueId()
        if rank == 0:
            self._check(lib.ncclGetUniqueId(ctypes.byref(uid)))
        t = torch.tensor(list(bytes(uid.internal)), dtype = torch.uint8, device = device)
        dist.broadcast(t, src = 0)
        ctypes.memmove(uid.internal, bytes(t.cpu().tolist()), 128)
        comm = ctypes.c_void_p()
        with torch.cuda.device(device):
            self._check(lib.ncclCommInitRank(ctypes.byref(comm), world, uid, rank))
        return comm

    def _check(self, rc: int):
        if rc != 0:
            raise RuntimeError(f"RCCL error {rc}: {self.lib.ncclGetErrorString(rc).decode()}")

    def all_reduce(self, t: torch.Tensor, op: str = "sum"):
        """In place, on the current stream."""
        assert t.is_contiguous() and t.device == self.device
        comm = self.comm_simple if self.comm_simple is not None and \
            t.numel() * t.element_size() >= self.simple_min else self.comm
        self._check(self.lib.ncclAllReduce(
            ctypes.c_void_p(t.data_ptr()), ctypes.c_void_p(t.data_ptr()), t.numel(), _DTYPES[t.dtype], _OPS[op],
            comm, ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)))
