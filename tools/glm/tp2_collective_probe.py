#!/usr/bin/env python3
"""Check two physical gfx1151 ranks using RCCL RoCE, without loading a model."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist


def run(output: Path, adapter_source: Path | None = None) -> None:
    rank = int(os.environ["RANK"])
    if rank not in (0, 1) or os.environ.get("WORLD_SIZE") != "2":
        raise ValueError("This check requires exactly two ranks")
    required = {"NCCL_NET": "IB", "NCCL_IB_DISABLE": "0", "NCCL_IB_ROCE_VERSION_NUM": "2"}
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise ValueError("Explicit RCCL IB / RoCE v2 settings are required")
    hca = os.environ["NCCL_IB_HCA"]
    gid_index = os.environ["NCCL_IB_GID_INDEX"]
    port = Path("/sys/class/infiniband") / hca / "ports/1"
    gid_type = (port / "gid_attrs/types" / gid_index).read_text().strip()
    gid = (port / "gids" / gid_index).read_text().strip()
    if gid_type != "RoCE v2" or gid == "0000:0000:0000:0000:0000:0000:0000:0000":
        raise ValueError("Selected GID must be populated RoCE v2")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    if not properties.gcnArchName.startswith("gfx1151"):
        raise ValueError("Expected gfx1151")
    report = {
        "scope": "two-host collective correctness smoke; no model or speed claim",
        "rank": rank, "host": socket.gethostname(), "local_device": 0,
        "torch": torch.__version__, "hip": torch.version.hip,
        "arch": properties.gcnArchName,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {key: value for key, value in sorted(os.environ.items())
                        if key.startswith(("NCCL_", "GLOO_")) or
                        key in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")},
        "gid": gid, "gid_type": gid_type, "checks": [], "complete": False,
    }
    try:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60), device_id=device)
        if adapter_source is not None:
            spec = importlib.util.spec_from_file_location("rank_collectives_probe", adapter_source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            backend = module.RankCollectives()
            reduce = backend.all_reduce
            broadcast = lambda value, src: backend.broadcast(value, src_device=src)
            report["adapter_sha256"] = hashlib.sha256(adapter_source.read_bytes()).hexdigest()
        else:
            reduce = dist.all_reduce
            broadcast = dist.broadcast
        hosts = [None, None]
        # Separate host identity from local GPU ordinal on each machine.
        dist.all_gather_object(hosts, socket.gethostname())
        if len(set(hosts)) != 2:
            raise ValueError("Ranks must run on distinct hosts")
        report["hosts"] = hosts
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for count in (1, 4096, 256 * 4096):
                value = torch.full((count,), rank + 1, dtype=dtype, device=device)
                reduce(value)
                torch.testing.assert_close(value, torch.full_like(value, 3), rtol=0, atol=0)
                report["checks"].append(f"sum:{dtype}:{count}")
        epsilon = 2.0 ** -17
        value = torch.tensor([1 + epsilon if rank == 0 else -1], dtype=torch.float32, device=device)
        reduce(value)
        torch.testing.assert_close(value, torch.full_like(value, epsilon), rtol=0, atol=0)
        report["checks"].append("float32-reduction-preserves-small-difference")
        value = torch.full((4096,), rank + 7, dtype=torch.int64, device=device)
        broadcast(value, src=1)
        torch.testing.assert_close(value, torch.full_like(value, 8), rtol=0, atol=0)
        report["checks"].append("broadcast-global-rank-one-local-device-zero")
        value = torch.full((4096,), rank, dtype=torch.float32, device=device)
        reduce(value)
        torch.testing.assert_close(value, torch.ones_like(value), rtol=0, atol=0)
        report["checks"].append("zero-contribution-participates-in-sum")
        if adapter_source is not None:
            value = torch.full((4096,), 17 if rank == 0 else 5, device=device)
            reduce(value, contribution=rank != 0)
            torch.testing.assert_close(value, torch.full_like(value, 5), rtol=0, atol=0)
            report["checks"].append("noncontributor-input-is-cleared")
            value = torch.full((17, 2), rank + 1, device=device)[:, 0]
            reduce(value)
            torch.testing.assert_close(value, torch.full_like(value, 3), rtol=0, atol=0)
            report["checks"].append("strided-reduction-copies-back")
        torch.cuda.synchronize(device)
        report["complete"] = True
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()
    print(json.dumps({"rank": rank, "complete": report["complete"], "checks": len(report["checks"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-source", type=Path)
    args = parser.parse_args()
    run(args.output, args.adapter_source)
