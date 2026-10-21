"""
Lockstep greedy generation across expert-parallel ranks (exllamav3/model/ep.py), one process per host.

Both ranks load the model with their own expert range, run the same prompts and must produce the same tokens:
the script compares every rank's token ids with rank 0's and exits non-zero on any difference.

    KYOJIN_EP_WORLD=2 KYOJIN_EP_RANK=<r> MASTER_ADDR=192.168.99.1 MASTER_PORT=29531 \
        python tools/glm/ep_generate.py --model <dir> [--max-new 128] [--ctx 8192]
"""
import argparse, os, sys, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "List the planets of the solar system in order from the Sun.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--max-new", type = int, default = 128)
    ap.add_argument("--ctx", type = int, default = 8192)
    ap.add_argument("--prompt", action = "append", default = None)
    ap.add_argument("--prefill-words", type = int, default = 0, help = "prepend a long filler document (prefill test)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    from exllamav3 import Cache, Config, Generator, Model, Tokenizer, GreedySampler
    from exllamav3.model.ep import ep_state

    ep = ep_state()
    rank = ep.rank if ep else 0
    t0 = time.time()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens = args.ctx + 4096, max_history = 1, max_batch_size = 1)
    model.load(device = "cuda:0", progressbar = rank == 0)
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"[rank {rank}] loaded in {time.time() - t0:.1f} s, device free {free / 2**30:.1f} / {total / 2**30:.1f} GiB",
          flush = True)
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer)

    prompts = args.prompt or PROMPTS
    filler = ""
    if args.prefill_words:
        words = ("the quick brown fox jumps over the lazy dog while seven wizards quietly judge boxing matches "
                 "near an old lighthouse ").split()
        filler = " ".join(words[i % len(words)] for i in range(args.prefill_words)) + "\n\n"

    ok = True
    for i, p in enumerate(prompts):
        text = f"[gMASK]<sop><|user|>{filler}{p}<|assistant|><think></think>"
        ids = tokenizer.encode(text, encode_special_tokens = True)
        stops = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        stops += [t for t in (154820, 154827, 154829) if t not in stops]
        torch.cuda.synchronize()
        t1 = time.time()
        out = generator.generate(
            prompt = text,
            max_new_tokens = args.max_new,
            sampler = GreedySampler(),
            encode_special_tokens = True,
            decode_special_tokens = False,
            stop_conditions = stops,
            completion_only = True,
            return_last_results = True,
        )
        torch.cuda.synchronize()
        dt = time.time() - t1
        completion, last = out
        new_ids = tokenizer.encode(completion, encode_special_tokens = False)
        nnew = int(last.get("new_tokens", new_ids.shape[-1])) if isinstance(last, dict) else new_ids.shape[-1]
        tps_dec = last.get("time_generate") if isinstance(last, dict) else None
        digest = hashlib.sha256(completion.encode()).hexdigest()[:16]
        stats = ""
        if isinstance(last, dict):
            pf_t = last.get("time_prefill"); gen_t = last.get("time_generate")
            pf_n = last.get("prompt_tokens") or ids.shape[-1]
            if pf_t and gen_t:
                stats = f" prefill {pf_n / pf_t:.1f} t/s ({pf_n} tok), decode {nnew / gen_t:.2f} t/s"
        print(f"[rank {rank}] prompt {i}: {nnew} tokens in {dt:.2f} s{stats} sha {digest}", flush = True)
        if rank == 0:
            print("-" * 60 + f"\n{completion}\n" + "-" * 60, flush = True)
        if ep:
            ref = ep.broadcast_obj(completion if rank == 0 else None, src = 0)
            same = ref == completion
            allsame = torch.tensor([1 if same else 0], device = "cuda")
            torch.distributed.all_reduce(allsame, op = torch.distributed.ReduceOp.MIN)
            if not bool(allsame.item()):
                ok = False
                print(f"[rank {rank}] MISMATCH with rank 0 on prompt {i}", flush = True)
            elif rank == 0:
                print(f"[rank 0] prompt {i}: ranks agree", flush = True)
    if ep:
        print(f"[rank {rank}] EP all-reduces: {ep.reduces}", flush = True)
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
