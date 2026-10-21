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
import threading
from datetime import timedelta
import torch
import torch.distributed as dist

_STATE = None


class EPDesync(RuntimeError):
    """The ranks are about to run different work (the collective schedules would no longer match)."""


class EPState:

    def __init__(self, rank: int, world: int):
        self.rank = rank
        self.world = world
        self.reduces = 0
        self.bf16_min = int(os.environ.get("KYOJIN_EP_BF16_MIN", 64 * 4096))
        self.share = float(os.environ.get("KYOJIN_EP_SHARE", "0.5"))
        # Two-batch overlap of prefill chunks (prefill_overlapped): the active hand-over, and which half this thread runs
        self.tbo = None
        self._tls = threading.local()
        self.tbo_min = int(os.environ.get("KYOJIN_EP_TBO_MIN", 1024))
        # KYOJIN_EP_REDUCE_PROF=1: synchronize around every all-reduce and report wall time per row count at exit
        self.prof = {} if os.environ.get("KYOJIN_EP_REDUCE_PROF", "0") != "0" else None
        if self.prof is not None:
            import atexit
            atexit.register(self.prof_report)

    def prof_report(self):
        for rows, (n, t) in sorted(self.prof.items()):
            print(f" -- EP reduce prof rank {self.rank}: rows {rows:5d}  calls {n:7d}  total {t * 1e3:10.1f} ms  "
                  f"mean {t / n * 1e6:8.1f} us", flush = True)

    def split(self, total: int, align: int = 1) -> tuple[int, int]:
        """(first, count) of this rank's share of total units (experts, heads, channels, vocab), in align-sized
        blocks. KYOJIN_EP_SHARE (two ranks; default 0.5) is rank 0's share: hosts of unequal sustained compute get
        work in proportion so neither waits at every all-reduce (heterogeneity-aware placement)."""
        assert total % align == 0
        blocks = total // align
        if self.world == 2:
            b0 = min(blocks - 1, max(1, round(blocks * self.share)))
            first, count = (0, b0) if self.rank == 0 else (b0, blocks - b0)
        else:
            assert blocks % self.world == 0, f"{total} not divisible into {self.world} x {align}"
            count = blocks // self.world
            first = self.rank * count
        return first * align, count * align

    def expert_range(self, num_experts: int) -> tuple[int, int]:
        first, count = self.split(num_experts)
        return first, first + count

    def expert_perm(self, num_experts: int) -> list[int]:
        """Router id order with this rank's experts first: perm[slot] = router expert id."""
        first, last = self.expert_range(num_experts)
        return list(range(first, last)) + [e for e in range(num_experts) if not first <= e < last]

    # Same signature as the TP backends' all_reduce (contribution=False: this rank adds zeros)
    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        if self.prof is not None:
            import time
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self._all_reduce(tensor, contribution)
            torch.cuda.synchronize()
            rows = tensor.numel() // tensor.shape[-1]
            n, t = self.prof.get(rows, (0, 0.0))
            self.prof[rows] = (n + 1, t + time.perf_counter() - t0)
            return
        tbo = self.tbo
        if tbo is not None and getattr(self._tls, "half", None) is not None:
            self._all_reduce_overlapped(tensor, contribution, tbo)
            return
        self._all_reduce(tensor, contribution)

    def _all_reduce_overlapped(self, tensor: torch.Tensor, contribution: bool, tbo):
        # Same arithmetic as _all_reduce, but the collective runs on RCCL's stream while the other half of the
        # prefill chunk computes: enqueue, hand over, then make the compute stream wait for the result
        if not contribution:
            tensor.zero_()
        if self.bf16_min and tensor.dtype == torch.float and tensor.numel() >= self.bf16_min:
            t = tensor.to(torch.bfloat16)
        elif tensor.is_contiguous():
            t = tensor
        else:
            t = tensor.contiguous()
        work = dist.all_reduce(t, async_op = True)
        tbo.switch(self._tls.half)
        work.wait()
        if t is not tensor:
            tensor.copy_(t)
        self.reduces += 1

    def _all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        if not contribution:
            tensor.zero_()
        # Large fp32 payloads (prefill chunks) go over the wire as bf16: half the bytes on a ~2 GB/s link.
        # Every rank gets the same rounded sum, so the ranks stay identical. KYOJIN_EP_BF16_MIN=0 disables
        if self.bf16_min and tensor.dtype == torch.float and tensor.numel() >= self.bf16_min:
            t = tensor.to(torch.bfloat16)
            dist.all_reduce(t)
            tensor.copy_(t)
            self.reduces += 1
            return
        if tensor.is_contiguous():
            dist.all_reduce(tensor)
        else:
            t = tensor.contiguous()
            dist.all_reduce(t)
            tensor.copy_(t)
        self.reduces += 1

    def split_counts(self, total: int, align: int = 1) -> list[int]:
        """Every rank's count under split() (rank order)."""
        me = self.rank
        counts = []
        for r in range(self.world):
            self.rank = r
            counts.append(self.split(total, align)[1])
        self.rank = me
        return counts

    def all_gather_last(self, t: torch.Tensor, counts: list[int] | None = None) -> torch.Tensor:
        """Concatenate every rank's t along the last dim in rank order (vocab-split head logits). counts: each
        rank's width when they differ (weighted split); shorter shares are padded on the wire and trimmed."""
        t = t.contiguous()
        w = max(counts) if counts else t.shape[-1]
        if t.shape[-1] != w:
            t = torch.nn.functional.pad(t, (0, w - t.shape[-1]))
        out = torch.empty((self.world,) + tuple(t.shape), dtype = t.dtype, device = t.device)
        dist.all_gather_into_tensor(out.view(-1), t.view(-1))
        if counts and min(counts) != w:
            return torch.cat([out[r, ..., :c] for r, c in enumerate(counts)], dim = -1)
        return out.movedim(0, -2).reshape(*t.shape[:-1], self.world * t.shape[-1])

    def broadcast_obj(self, obj = None, src: int = 0):
        box = [obj]
        dist.broadcast_object_list(box, src = src)
        return box[0]

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        dist.broadcast(tensor, src = src)

    def barrier(self):
        dist.barrier()


class _HandOver:
    """Strict alternation of two threads: exactly one runs Python/launches at a time and control passes only at
    all-reduces, so the launch order (and the collective order, which must match the peer's) is deterministic."""

    def __init__(self):
        self.cv = threading.Condition()
        self.turn = 0
        self.done = [False, False]

    def wait_turn(self, me: int):
        with self.cv:
            self.cv.wait_for(lambda: self.turn == me)

    def switch(self, me: int):
        with self.cv:
            if self.done[1 - me]:
                return
            self.turn = 1 - me
            self.cv.notify_all()
            self.cv.wait_for(lambda: self.turn == me)

    def finish(self, me: int):
        with self.cv:
            self.done[me] = True
            self.turn = 1 - me
            self.cv.notify_all()


PAGE = 256


def prefill_overlapped(model, input_ids: torch.Tensor, params: dict) -> bool:
    """
    Two-batch overlap (SGLang TBO) for one EP prefill chunk: split the rows at a page boundary into halves A and B
    and run both forwards interleaved on the same stream, A's all-reduce in flight while B computes and vice versa.
    B runs each layer after A's same layer, so it sees A's K/V pages and A's recurrent (KDA) state exactly as a
    sequential second chunk would. Returns False (caller runs the chunk whole) when the chunk is not eligible.
    KYOJIN_EP_TBO=0 disables; KYOJIN_EP_TBO_MIN (default 1024) is the smallest chunk that is split.
    """
    ep = ep_state()
    if ep is None or ep.prof is not None or os.environ.get("KYOJIN_EP_TBO", "1") == "0":
        return False
    if input_ids.dim() != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < ep.tbo_min:
        return False
    cache_seqlens = params.get("cache_seqlens")
    if cache_seqlens is None or "batch_shape" in params or params.get("indexed_embeddings") or \
            params.get("mm_span_prefix") or params.get("kv_only") or params.get("recurrent_history") or \
            params.get("non_causal_spans"):
        return False
    n = input_ids.shape[1]
    start = int(cache_seqlens[0])
    rows = [int(r) for r in params.get("checkpoint_rows") or []]
    # page-aligned split near the middle, never on a checkpoint row (a half cannot side-write its first/last row)
    cands = [p - start for p in range((start // PAGE + 1) * PAGE, start + n, PAGE) if p - start not in rows]
    cands = [c for c in cands if 256 <= c <= n - 256]
    if not cands:
        return False
    na = min(cands, key = lambda c: abs(c - n / 2))

    halves = []
    for lo, hi in ((0, na), (na, n)):
        p = dict(params)
        p["cache_seqlens"] = cache_seqlens + lo
        if rows:
            r = [x - lo for x in rows if lo < x < hi]
            if r:
                p["checkpoint_rows"] = r
            else:
                p.pop("checkpoint_rows", None)
                p.pop("midchunk_mode", None)
        ids = input_ids[:, lo:hi]
        x = model.prepare_inputs(ids, p)
        halves.append((x, p))

    tbo = _HandOver()
    errors = [None, None]
    device = torch.cuda.current_device()

    def run(i):
        ep._tls.half = i
        try:
            torch.cuda.set_device(device)
            with torch.inference_mode():
                tbo.wait_turn(i)
                for m in model._get_prefetch_layers:
                    m.prefetch(*halves[i])
                model.prefill_ls(*halves[i])
        except BaseException as e:
            errors[i] = e
        finally:
            ep._tls.half = None
            tbo.finish(i)

    dbg = os.environ.get("KYOJIN_EP_TBO_DEBUG", "0") != "0"
    if dbg:
        import time
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.cuda.set_sync_debug_mode("warn")
    ep.tbo = tbo
    try:
        tb = threading.Thread(target = run, args = (1,), name = "ep-tbo-B")
        tb.start()
        run(0)
        tb.join()
    finally:
        ep.tbo = None
    if dbg:
        torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        print(f" -- EP TBO prefill rows {n} split {na}+{n - na} at {start}: {(time.perf_counter() - t0) * 1e3:.1f} ms", flush = True)
    for e in errors:
        if e is not None:
            raise e

    (_, pa), (_, pb) = halves
    if pa.get("export_states"):
        params["export_states"] = [torch.cat((a, b), dim = -2) for a, b in zip(pa["export_states"], pb["export_states"])]
    mid = dict(pa.get("midchunk_states") or {})
    mid.update({r + na: v for r, v in (pb.get("midchunk_states") or {}).items()})
    if mid:
        params["midchunk_states"] = mid
    return True


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
        # A collective that one rank never joins aborts the process after this long (torch's NCCL watchdog,
        # TORCH_NCCL_ASYNC_ERROR_HANDLING=3) instead of hanging the pair forever
        timeout = timedelta(seconds = _env_int("KYOJIN_EP_NCCL_TIMEOUT", default = 300))
        dist.init_process_group(
            "nccl",
            init_method = f"tcp://{addr}:{port}",
            rank = rank,
            world_size = world,
            device_id = device,
            timeout = timeout,
        )
        # First collective sets up the RCCL communicator; fail loudly here, not inside a layer
        t = torch.full((4096,), float(rank + 1), device = device)
        dist.all_reduce(t)
        expect = world * (world + 1) / 2
        assert float(t[0]) == expect, f"EP all-reduce self-test failed: {float(t[0])} != {expect}"
        # Settings that change what each rank computes must be identical everywhere (a different share or Q8 flag
        # on one host gives wrong output without any desync): compare them once, fail before loading the model
        names = ("KYOJIN_EP_SHARE", "KYOJIN_Q8_LINEAR", "KYOJIN_Q8_MIN_NUMEL", "KYOJIN_EP_TBO", "KYOJIN_EP_TBO_MIN",
                 "KYOJIN_EP_BF16_MIN", "KYOJIN_EP_KDA_SPLIT", "KYOJIN_EP_MLA_SPLIT", "KYOJIN_EP_DENSE_SPLIT",
                 "KYOJIN_EP_SHARED_SPLIT", "KYOJIN_EP_HEAD_SPLIT", "KYOJIN_EP_DRAFT_SPLIT", "KYOJIN_Q8_SKIP")
        import zlib
        cfg = torch.tensor([float(zlib.crc32(os.environ.get(k, "").encode())) for k in names],
                           dtype = torch.float64, device = device)
        hi, lo = cfg.clone(), -cfg
        dist.all_reduce(hi, op = dist.ReduceOp.MAX)
        dist.all_reduce(lo, op = dist.ReduceOp.MAX)
        bad = [k for k, h, l in zip(names, hi.tolist(), lo.tolist()) if h != -l]
        if bad:
            raise EPDesync(f"rank {rank}: EP settings differ across hosts: {', '.join(bad)} (set them identically)")
        torch.cuda.synchronize()
    _STATE = EPState(rank, world)
    print(f" -- EP rank {rank}/{world} ready (torch.distributed nccl)", flush = True)
    return _STATE
