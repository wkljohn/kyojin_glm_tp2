#!/usr/bin/env python3
"""Ordinary GLM EXL3 generation with one channel-sharded rank per physical host."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist
from jinja2.sandbox import ImmutableSandboxedEnvironment

from exllamav3 import Cache, Config, Model, Tokenizer
from exllamav3.constants import PAGE_SIZE
from exllamav3.ext import exllamav3_ext
from exllamav3.model.glm_channel_tp import configure_glm_channel_tp
from exllamav3.model.rank_collectives import RankCollectives


@torch.inference_mode()
def run(args):
    if args.context < PAGE_SIZE or args.context % PAGE_SIZE or args.prefill_chunk < 1 or args.max_new_tokens < 1:
        raise ValueError(f"Positive generation limits and context divisible by {PAGE_SIZE} are required")
    if os.environ.get("EXL3_HAD_PF_FAST", "0") != "0":
        raise ValueError("Expanded-weight caching is outside this experiment")
    if os.environ.get("NCCL_NET") != "IB" or os.environ.get("NCCL_IB_DISABLE") != "0":
        raise ValueError("Use explicit NCCL/RCCL IB transport for this experiment")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=300), device_id=torch.device("cuda", 0))
    backend = RankCollectives()
    report = {
        "scope": "ordinary GLM EXL3 channel TP, replicated attention, no speculation",
        "rank": backend.rank, "host": socket.gethostname(), "model": str(args.model.resolve()),
        "config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "hip": torch.version.hip,
        "extension_sha256": hashlib.sha256(Path(exllamav3_ext.__file__).read_bytes()).hexdigest(),
        "limits": {"context": args.context, "prefill_chunk": args.prefill_chunk,
                   "max_new_tokens": args.max_new_tokens, "no_thinking": args.no_thinking},
        "environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "GLOO_", "EXL3_"))},
        "requests": [], "complete": False,
    }
    output = args.output.with_name(args.output.name + f".rank{backend.rank}.json")
    try:
        hosts = [None, None]
        dist.all_gather_object(hosts, socket.gethostname())
        if len(set(hosts)) != 2:
            raise ValueError("This launcher requires two physical hosts")
        report["hosts"] = hosts
        config = Config.from_directory(str(args.model))
        model = Model.from_config(config)
        report["sharded_moe_layers"] = configure_glm_channel_tp(model, backend)
        cache = Cache(model, max_num_tokens=args.context, max_batch_size=1, max_history=0)
        tokenizer = Tokenizer.from_config(config)
        template_path = args.model / "chat_template.jinja"
        template_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                     extensions=["jinja2.ext.loopcontrols"])
        template_env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, **kwargs)
        template = template_env.from_string(template_path.read_text())
        report["template_sha256"] = hashlib.sha256(template_path.read_bytes()).hexdigest()

        def loaded(index, total):
            print(json.dumps({"rank": backend.rank, "loading_module": index, "total": total}), flush=True)

        model.load(device="cuda:0", callback=loaded)
        report["allocated_weight_and_cache_bytes"] = torch.cuda.memory_allocated()
        dist.barrier()
        previous_outputs = {}
        for prompt in args.prompt:
            text = template.render(messages=[{"role": "user", "content": prompt}],
                                   tools=[], add_generation_prompt=True, reasoning_effort="low")
            if args.no_thinking:
                text += "</think>\n"
            ids = tokenizer.encode(text, encode_special_tokens=True)
            prompt_length = ids.shape[-1]
            if prompt_length + args.max_new_tokens > args.context:
                raise ValueError("Prompt and requested output exceed the allocated context")
            states = None
            generated = []
            started = time.monotonic()
            try:
                for offset in range(0, max(0, prompt_length - 1), args.prefill_chunk):
                    end = min(prompt_length - 1, offset + args.prefill_chunk)
                    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": offset,
                              "batch_shape": (1, args.context), "recurrent_states": states}
                    model.prefill(ids[:, offset:end], params)
                    states = params.get("recurrent_states")
                token = ids[:, -1:]
                for step in range(args.max_new_tokens):
                    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": prompt_length - 1 + step,
                              "batch_shape": (1, args.context), "recurrent_states": states}
                    logits = model.forward(token, params)
                    states = params.get("recurrent_states")
                    finite = torch.isfinite(logits).all().to(torch.int32)
                    backend.all_reduce(finite)
                    if finite.item() != 2:
                        raise RuntimeError("A rank produced nonfinite logits")
                    next_token = logits[:, -1, :config.vocab_size].argmax(dim=-1).reshape(1, 1)
                    local_choice = next_token.clone()
                    backend.broadcast(next_token, src_device=0)
                    agreement = torch.eq(local_choice, next_token).all().to(torch.int32)
                    backend.all_reduce(agreement)
                    if agreement.item() != 2:
                        raise RuntimeError("The two ranks disagree on the next greedy token")
                    item = next_token.item()
                    generated.append(item)
                    if backend.rank == 0:
                        print(json.dumps({"step": step, "token": item}), flush=True)
                    if item in config.eos_token_id_list:
                        break
                    token = next_token.cpu()
            finally:
                if states:
                    for state in states:
                        state.free()
            record = {"prompt": prompt, "prompt_tokens": prompt_length, "generated_tokens": generated,
                      "text": tokenizer.decode(torch.tensor([generated]))[0],
                      "elapsed_seconds": time.monotonic() - started,
                      "repeated_request_match": None}
            if prompt in previous_outputs:
                record["repeated_request_match"] = previous_outputs[prompt] == generated
                if not record["repeated_request_match"]:
                    raise AssertionError("Greedy output changed after independent request reset")
            previous_outputs[prompt] = list(generated)
            report["requests"].append(record)
            if backend.rank == 0:
                print(json.dumps(record, ensure_ascii=False), flush=True)
        report["complete"] = True
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--no-thinking", action="store_true")
    run(parser.parse_args())
