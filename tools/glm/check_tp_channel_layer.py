#!/usr/bin/env python3
"""Compare channel shards with an unsplit layer of the selected real EXL3 pack.

This is a single-GPU arithmetic/loading component oracle. The rank backend
captures each partial locally; use tp2_collective_probe.py for real transport.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from exllamav3 import Config, Model
from exllamav3.model.glm_channel_tp import configure_glm_channel_tp


class _LocalPartial:
    world_size = 2

    def __init__(self, rank):
        self.rank = rank
        self.reductions = 0

    def broadcast(self, value, src_device):
        if src_device != 0:
            raise ValueError("Unexpected routing owner")

    def all_reduce(self, value, contribution=True):
        self.reductions += 1
        if not contribution:
            value.zero_()


@torch.inference_mode()
def run(model_path: Path, output: Path, layer: int):
    config = Config.from_directory(str(model_path))
    key = f"model.language_model.layers.{layer}.mlp"
    backends = [_LocalPartial(rank) for rank in range(2)]
    models = [Model.from_config(config) for _ in range(3)]
    for rank in range(2):
        configure_glm_channel_tp(models[rank + 1], backends[rank])
    modules = [model.find_module(key) for model in models]
    for module in modules:
        module.load(torch.device("cuda", 0))
    reference, rank0, rank1 = modules
    if not (rank0.intermediate_size == rank1.intermediate_size == 1024):
        raise AssertionError("Expected two 1024-channel shards")
    if rank0.shared_experts is None or rank1.shared_experts is not None:
        raise AssertionError("Exactly one shared-expert contribution is required")
    checked = 0
    for name, axis in (("gates", 1), ("ups", 1), ("downs", 0)):
        for full, a, b in zip(getattr(reference, name), getattr(rank0, name), getattr(rank1, name), strict=True):
            if full.quant_type != "exl3":
                raise AssertionError("This pack's routed experts must remain EXL3")
            full, a, b = full.inner, a.inner, b.inner
            if not torch.equal(torch.cat((a.trellis, b.trellis), dim=axis), full.trellis):
                raise AssertionError("Packed expert trellis changed during slicing")
            for vector, split in (("suh", axis == 0), ("svh", axis == 1)):
                actual = torch.cat((getattr(a, vector), getattr(b, vector))) if split else getattr(a, vector)
                if not torch.equal(actual, getattr(full, vector)):
                    raise AssertionError("EXL3 sign vector changed during slicing")
                if not split and not torch.equal(getattr(b, vector), getattr(full, vector)):
                    raise AssertionError("Replicated sign vector mismatch")
            if not (full.mcg == a.mcg == b.mcg and full.mul1 == a.mul1 == b.mul1):
                raise AssertionError("EXL3 codebook changed during slicing")
            checked += 1
    report = {
        "scope": "single-GPU real-weight channel-sharding component oracle",
        "model": str(model_path.resolve()), "layer": layer,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
        "packed_projections_exact": checked, "rows": [], "complete": False,
        "limits": {"relative_l2": 0.005, "max_error_over_peak": 0.02},
    }
    try:
        for rows in (1, 2, 8, 33):
            torch.manual_seed(5353 + rows)
            x = torch.randn((1, rows, 4096), device="cuda", dtype=torch.float16) * 0.1
            selected = torch.tensor([0, 1, 127, 143, 144, 255, 286, 287], device="cuda").repeat(rows, 1)
            weights = torch.full((rows, 8), 1 / 8, dtype=torch.float16, device="cuda")
            expected = reference.forward(x, {"dec_routed": (selected, weights)}).float().clone()
            partials = [module.forward(x, {"dec_routed": (selected, weights), "backend": backend}).float().clone()
                        for module, backend in zip((rank0, rank1), backends, strict=True)]
            actual = partials[0] + partials[1]
            if not (torch.isfinite(expected).all() and torch.isfinite(actual).all()):
                raise AssertionError("Nonfinite component output")
            error = actual - expected
            relative_l2 = (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected).clamp_min(1e-12)).item()
            peak_error = (error.abs().max() / expected.abs().max().clamp_min(1e-12)).item()
            report["rows"].append({"count": rows, "relative_l2": relative_l2, "max_error_over_peak": peak_error,
                                   "reference_peak": expected.abs().max().item()})
            if relative_l2 > report["limits"]["relative_l2"] or peak_error > report["limits"]["max_error_over_peak"]:
                raise AssertionError(f"Channel sum disagrees with full real-weight MoE at {rows} rows")
        if any(backend.reductions != len(report["rows"]) for backend in backends):
            raise AssertionError("Every partial must reach exactly one parent reduction")
        report["complete"] = True
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    args = parser.parse_args()
    run(args.model, args.output, args.layer)
