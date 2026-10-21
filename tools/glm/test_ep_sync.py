"""
CPU test of the cross-host EP lockstep guards in Generator (no GPU, no model): two gloo processes on localhost.

    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= .venv/bin/python tools/glm/test_ep_sync.py

1. hazard: two calibrators with the same history, fed confidences one ulp apart at a bin edge, pick different draft
   widths (8 verify rows on one rank, 1 on the other: the hang seen in production)
2. _ep_share_draft: rank-local drafts that differ come out identical (rank 0's) on both ranks, ids/conf/len exact
3. _ep_check_rows: same rows pass; a different width, batch or draft token raises EPDesync on both ranks
"""
import os, sys, math
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def width(cal, conf, window):
    # generator.iterate_draftmodel_dflash_gen's truncation, batch of one
    below = conf[:, :window] < cal.threshold()
    cuts = torch.where(below.any(dim = 1), below.int().argmax(dim = 1), torch.full((1,), window))
    return int(cuts.max())


def hazard():
    from exllamav3.generator.draft_confidence import DraftConfidenceCalibrator
    cals = [DraftConfidenceCalibrator(0.4) for _ in range(2)]
    for cal in cals:
        for _ in range(200):
            cal.add_label(5.5, True)       # bin 5: accepted
            cal.add_label(4.5, False)      # bin 4: rejected
    thr = cals[0].threshold()
    assert thr == cals[1].threshold() == 5.0, thr
    edge = torch.tensor(5.0, dtype = torch.float32)
    lo = torch.nextafter(edge, torch.tensor(0.0))
    c0 = torch.full((1, 7), 6.0); c0[0, 0] = edge
    c1 = c0.clone(); c1[0, 0] = lo
    w0, w1 = width(cals[0], c0, 7), width(cals[1], c1, 7)
    print(f"[hazard] threshold {thr}; conf {edge.item()!r} vs {lo.item()!r} (1 ulp) -> width {w0} vs {w1} "
          f"({w0 + 1} vs {1 if w1 == 0 else w1 + 1} verify rows)")
    assert w0 == 7 and w1 == 0


def worker(rank, port, q):
    try:
        os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
        dist.init_process_group("gloo", rank = rank, world_size = 2)
        from exllamav3.generator.generator import Generator
        from exllamav3.model.ep import EPDesync
        g = SimpleNamespace(_ep = SimpleNamespace(rank = rank))
        out = {}

        # 2. draft sharing
        ids = torch.tensor([[11, 151000 + rank, 33, 44, 55, 66, 77, 88]])
        conf = torch.tensor([[0.0, 5.0 - rank * 1e-6, 4.25, 3.5, 2.75, 2.0, 1.25, 0.5]], dtype = torch.float16).float()
        params = {"draft_conf": conf.clone(), "draft_confidence_len": 7 - rank}
        new = Generator._ep_share_draft(g, ids.clone(), params)
        out["ids"] = new.tolist(); out["ids_dtype"] = str(new.dtype)
        out["conf"] = params["draft_conf"].tolist(); out["len"] = params["draft_confidence_len"]

        # 3. row guard
        def check(b, d):
            try:
                Generator._ep_check_rows(g, b, d)
                return "ok"
            except EPDesync:
                return "desync"
        d = torch.tensor([[5, 6, 7]])
        out["same"] = check(1, d)
        out["none"] = check(1, None)
        out["width"] = check(1, d if rank == 0 else None)
        out["batch"] = check(1 + rank, d)
        out["token"] = check(1, d if rank == 0 else torch.tensor([[5, 6, 8]]))
        q.put((rank, out))
        dist.destroy_process_group()
    except Exception as e:
        q.put((rank, repr(e)))


def main():
    hazard()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29000 + os.getpid() % 900
    ps = [ctx.Process(target = worker, args = (r, port, q)) for r in range(2)]
    for p in ps: p.start()
    res = dict(q.get(timeout = 300) for _ in ps)
    for p in ps: p.join(60)
    r0, r1 = res[0], res[1]
    assert isinstance(r0, dict) and isinstance(r1, dict), res
    assert r0["ids"] == r1["ids"] == [[11, 151000, 33, 44, 55, 66, 77, 88]], (r0["ids"], r1["ids"])
    assert r0["ids_dtype"] == r1["ids_dtype"] == "torch.int64"
    assert r0["conf"] == r1["conf"] and r0["len"] == r1["len"] == 7
    print(f"[share] rank-local drafts differed (id, conf, len); both ranks now use rank 0's: ids {r1['ids'][0][:2]}, "
          f"conf[1] {r1['conf'][0][1]}, len {r1['len']}")
    for k, want in (("same", "ok"), ("none", "ok"), ("width", "desync"), ("batch", "desync"), ("token", "desync")):
        assert r0[k] == r1[k] == want, (k, r0[k], r1[k])
    print("[guard] same rows ok on both; width / batch / draft-token mismatch -> EPDesync on both ranks")
    print("PASS")


if __name__ == "__main__":
    main()
