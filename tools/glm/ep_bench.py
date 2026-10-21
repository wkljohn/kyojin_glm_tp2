"""
GLM-5.3-Flash speed and agreement benchmark for one host or the two-host expert-parallel layout (ep.py).

Modelled on ds4-bench: fixed-length prompts, a fixed number of generated tokens, prefill and decode measured
separately, one CSV row per case. Unlike a single long document, the prompts cover eight different families so
routing, speculative acceptance and kernels are not tuned against one style of text:

  dossier   mixed technical records (the DS4 cross-discipline prompt)      english, facts + numbers
  code      Kyojin's own Python sources, review request                    code
  prose     I Promessi Sposi                                                italian literature
  chat      multi-turn assistant conversation                               dialogue
  math      arithmetic / algebra word problems, solve the last one         reasoning
  json      tabular JSON records, aggregation request                       structured data
  zh        Chinese operations log                                          chinese
  needle    filler document with one hidden fact, retrieval question        long-context recall

Every prompt is cut to exactly --prompt-tokens tokens (chat wrapper included) and generation is forced to exactly
--gen-tokens tokens (greedy, stop conditions off), so prefill and decode rates are comparable across cases and
runs. Under EP both ranks run every case and the generated token ids are compared; any difference fails the run.

    tools/glm/ep_launch.sh tools/glm/ep_bench.py --model <pack> --out <dir> [--prompt-tokens 2048] [--gen-tokens 256]
"""
import argparse, csv, hashlib, json, os, random, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "bench_prompts"
ROOT = HERE.parent.parent
FAMILIES = ["dossier", "code", "prose", "chat", "math", "json", "zh", "needle"]


def _rng(name):
    return random.Random(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))


def body_dossier():
    return (PROMPTS / "dossier.md").read_text(), \
        "Using only the records above, list the three most important unresolved risks, each with the exact numbers that support it."


def body_code():
    files = ["exllamav3/modules/rmsnorm.py", "exllamav3/modules/block_sparse_mlp_routing.py",
             "exllamav3/modules/linear.py", "exllamav3/cache/cache.py"]
    text = "".join(f"\n# ===== {f} =====\n" + (ROOT / f).read_text() for f in files if (ROOT / f).exists())
    return text, "Review the code above. Name the two most likely bugs or fragile assumptions and propose a fix for each, with code."


def body_prose():
    return (PROMPTS / "italian_prose.txt").read_text(), \
        "Riassumi in italiano il brano precedente in cinque frasi e spiega lo stile dell'autore."


def body_chat():
    r = _rng("chat")
    topics = ["planning a hiking trip in the Dolomites", "debugging a flaky CI job", "choosing a used electric car",
              "learning to bake sourdough", "migrating a Postgres database", "training for a half marathon",
              "writing a grant proposal", "setting up a home network with VLANs"]
    turns = []
    for i in range(80):
        t = topics[i % len(topics)]
        n = r.randint(2, 9)
        turns.append(f"<|user|>Question {i + 1} about {t}: I tried option {n} but it took {r.randint(3, 90)} minutes "
                     f"and cost {r.randint(5, 400)} euros. What should I change next time?")
        turns.append(f"<|assistant|><think></think>For {t}, option {n} is reasonable, but reduce the setup time by preparing "
                     f"step {r.randint(1, 5)} in advance, compare at least {r.randint(2, 4)} alternatives, and keep a short "
                     f"log so the next attempt starts from measured facts rather than memory.")
    return "".join(turns), "Looking back over this whole conversation, which three pieces of advice did I receive most often?"


def body_math():
    r = _rng("math")
    lines = []
    for i in range(400):
        a, b, c = r.randint(12, 999), r.randint(3, 97), r.randint(2, 40)
        kind = i % 4
        if kind == 0:
            lines.append(f"Problem {i + 1}: A tank holds {a} litres and drains {b} litres per hour while a pump adds {c} litres per hour. After how many hours is it empty?")
        elif kind == 1:
            lines.append(f"Problem {i + 1}: Solve for x: {b}x + {a} = {c}x + {a + b * c}.")
        elif kind == 2:
            lines.append(f"Problem {i + 1}: A shop discounts an item of {a} euros by {b % 50} percent, then adds {c} percent tax. What is the final price?")
        else:
            lines.append(f"Problem {i + 1}: The sum of three consecutive integers is {3 * a + 3}. What is the largest?")
    return "\n".join(lines), "Solve the last problem in the list step by step and state the final answer clearly."


def body_json():
    r = _rng("json")
    cities = ["Lisbon", "Osaka", "Nairobi", "Lima", "Oslo", "Hanoi", "Denver", "Tunis"]
    rows = [json.dumps({"id": i, "city": cities[r.randrange(len(cities))], "sensor": f"S{r.randint(100, 999)}",
                        "temp_c": round(r.uniform(-5, 38), 1), "humidity": r.randint(10, 99),
                        "status": r.choice(["ok", "ok", "ok", "drift", "offline"])}) for i in range(600)]
    return "\n".join(rows), ("Return a JSON object mapping each city to {\"mean_temp_c\", \"offline_count\"} computed from the records "
                             "above, then one sentence on data quality.")


def body_zh():
    r = _rng("zh")
    svc = ["支付网关", "订单服务", "库存系统", "用户中心", "日志平台", "推荐引擎"]
    lines = []
    for i in range(400):
        lines.append(f"记录{i + 1}：{svc[i % len(svc)]}在第{r.randint(1, 30)}天出现延迟{r.randint(5, 900)}毫秒，"
                     f"重试{r.randint(0, 6)}次，影响请求{r.randint(10, 9999)}个，负责人已提交修复编号FX-{r.randint(100, 999)}。")
    return "\n".join(lines), "请用中文总结以上运维日志中最严重的三个问题，并给出改进建议。"


NEEDLE = "The vault access code chosen by the archivist Mirela Ostrova is 7419-KESTREL."


def body_needle():
    r = _rng("needle")
    words = ("river stone lantern market harbor quiet morning ledger copper window garden signal orchard bridge "
             "winter paper engine valley mirror cedar").split()
    sents = [" ".join(r.choice(words) for _ in range(14)).capitalize() + "." for _ in range(900)]
    return " ".join(sents), "What vault access code did the archivist choose, and who is the archivist? Answer in one sentence."


BODIES = dict(dossier = body_dossier, code = body_code, prose = body_prose, chat = body_chat, math = body_math,
              json = body_json, zh = body_zh, needle = body_needle)


def build_ids(tokenizer, family, n_tokens):
    """[gMASK]<sop><|user|>{body cut to fit}\n\n{question}<|assistant|><think></think> == n_tokens ids exactly."""
    body, question = BODIES[family]()
    head = tokenizer.encode("[gMASK]<sop><|user|>", encode_special_tokens = True)
    tail = tokenizer.encode(f"\n\n{question}<|assistant|><think></think>", encode_special_tokens = True)
    b = tokenizer.encode(body, encode_special_tokens = False)
    room = n_tokens - head.shape[-1] - tail.shape[-1]
    assert b.shape[-1] >= room, f"{family}: body has {b.shape[-1]} tokens, needs {room}"
    if family == "needle":
        # the fact sits at 40 % depth of the cut document
        fact = tokenizer.encode(" " + NEEDLE + " ", encode_special_tokens = False)
        b = b[:, :room - fact.shape[-1]]
        at = b.shape[-1] * 2 // 5
        b = torch.cat((b[:, :at], fact, b[:, at:]), dim = -1)
    elif family in ("math", "chat", "zh", "json"):
        b = b[:, -room:]   # list-like bodies: keep the tail so the last item is complete
    else:
        b = b[:, :room]
    ids = torch.cat((head, b, tail), dim = -1)
    assert ids.shape[-1] == n_tokens
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--out", required = True, help = "output directory (research root)")
    ap.add_argument("--prompt-tokens", type = int, default = 2048)
    ap.add_argument("--gen-tokens", type = int, default = 256)
    ap.add_argument("--families", default = ",".join(FAMILIES))
    ap.add_argument("--repeat", type = int, default = 1)
    ap.add_argument("--draft", default = None, help = "draft model directory (DFlash / DFlash2)")
    ap.add_argument("--num-draft", type = int, default = 0)
    ap.add_argument("--label", default = "")
    ap.add_argument("--dynamic-draft", action = "store_true", help = "calibrated confidence truncation of the draft window")
    ap.add_argument("--draft-confidence", type = float, default = 0.4)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer, GreedySampler
    from exllamav3.model.ep import ep_state

    ep = ep_state()
    rank = ep.rank if ep else 0
    world = ep.world if ep else 1
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents = True, exist_ok = True)

    t0 = time.time()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    ctx = args.prompt_tokens + args.gen_tokens + 1024
    hist = max(1, args.num_draft)
    cache = Cache(model, max_num_tokens = ctx + 4096, max_history = hist, max_batch_size = 1)
    model.load(device = "cuda:0", progressbar = False)
    draft_model = draft_cache = None
    if args.draft:
        dconfig = Config.from_directory(args.draft)
        draft_model = Model.from_config(dconfig)
        draft_cache = Cache(draft_model, max_num_tokens = ctx + 4096, max_history = hist, max_batch_size = 1)
        # Timing-driven speculation decisions differ per rank and would desynchronize the EP collectives
        assert os.environ.get("EXL3_SPEC_GATE", "0") == "0", "EXL3_SPEC_GATE must stay off under EP"
        draft_model.load(device = "cuda:0", progressbar = False)
    load_s = time.time() - t0
    free, total = torch.cuda.mem_get_info()
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer,
                          draft_model = draft_model, draft_cache = draft_cache,
                          num_draft_tokens = args.num_draft or None,
                          dynamic_draft_tokens = args.dynamic_draft, draft_confidence = args.draft_confidence)

    def run_case(ids):
        job = Job(input_ids = ids, max_new_tokens = args.gen_tokens, min_new_tokens = args.gen_tokens,
                  sampler = GreedySampler(), stop_conditions = [], decode_special_tokens = True)
        generator.enqueue(job)
        toks, last, text = [], None, []
        torch.cuda.synchronize()
        t = time.perf_counter()
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("stage") == "streaming":
                    if r.get("token_ids") is not None:
                        toks += r["token_ids"].flatten().tolist()
                    text.append(r.get("text", ""))
                    if r.get("eos"):
                        last = r
        torch.cuda.synchronize()
        return toks, last, "".join(text), time.perf_counter() - t

    # Warm-up: kernel JIT, allocator growth, first-use tuning (not reported)
    # full prompt length: the first measured case must not pay for new prefill shapes
    warm = build_ids(tokenizer, "math", args.prompt_tokens)
    run_case(warm)

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    rows, ok = [], True
    for rep in range(args.repeat):
        for fam in fams:
            ids = build_ids(tokenizer, fam, args.prompt_tokens)
            toks, last, text, wall = run_case(ids)
            pf_t, gen_t = last["time_prefill"], last["time_generate"]
            n_new = int(last["new_tokens"])
            acc, rej = last.get("accepted_draft_tokens", 0), last.get("rejected_draft_tokens", 0)
            digest = hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]
            agree = True
            if ep:
                ref = ep.broadcast_obj(toks if rank == 0 else None, src = 0)
                flag = torch.tensor([1 if ref == toks else 0], device = "cuda")
                torch.distributed.all_reduce(flag, op = torch.distributed.ReduceOp.MIN)
                agree = bool(flag.item())
                ok &= agree
            row = dict(label = args.label, family = fam, rep = rep, world = world,
                       prompt_tokens = int(last["prompt_tokens"]), gen_tokens = n_new,
                       prefill_s = round(pf_t, 4), decode_s = round(gen_t, 4),
                       prefill_tps = round(last["prompt_tokens"] / pf_t, 2) if pf_t else None,
                       decode_tps = round(n_new / gen_t, 3) if gen_t else None,
                       accepted = acc, rejected = rej,
                       accept_rate = round(acc / (acc + rej), 4) if acc + rej else None,
                       tokens_per_round = round(n_new / max(1, n_new - acc), 3) if acc + rej else None,
                       wall_s = round(wall, 3), sha = digest, ranks_agree = agree)
            rows.append(row)
            if rank == 0:
                print(f"[bench] {fam:8s} prefill {row['prefill_tps']:8.2f} t/s  decode {row['decode_tps']:7.3f} t/s"
                      + (f"  accept {row['accept_rate']}" if row["accept_rate"] is not None else "")
                      + f"  agree={agree}  sha {digest}", flush = True)
                (out / f"sample-{fam}.txt").write_text(text)

    if rank == 0:
        def mean(k):
            v = [r[k] for r in rows if r[k] is not None]
            return round(sum(v) / len(v), 3) if v else None
        summary = dict(label = args.label, model = args.model, draft = args.draft, num_draft = args.num_draft,
                       dynamic_draft = args.dynamic_draft, draft_confidence = args.draft_confidence,
                       world = world, prompt_tokens = args.prompt_tokens, gen_tokens = args.gen_tokens,
                       load_s = round(load_s, 1), device_free_gib = round(free / 2**30, 1),
                       prefill_tps_mean = mean("prefill_tps"), decode_tps_mean = mean("decode_tps"),
                       prefill_tps_min = min(r["prefill_tps"] for r in rows),
                       decode_tps_min = min(r["decode_tps"] for r in rows),
                       accept_rate_mean = mean("accept_rate"), all_ranks_agree = ok,
                       ep_all_reduces = ep.reduces if ep else 0,
                       env = {k: v for k, v in os.environ.items() if k.startswith(("KYOJIN_", "EXL3_", "NCCL_IB_", "NCCL_NET"))},
                       time = time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(out / "bench.csv", "w", newline = "") as f:
            w = csv.DictWriter(f, fieldnames = list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        (out / "summary.json").write_text(json.dumps(summary, indent = 2))
        print("[bench] summary " + json.dumps({k: summary[k] for k in (
            "prefill_tps_mean", "decode_tps_mean", "prefill_tps_min", "decode_tps_min", "accept_rate_mean", "all_ranks_agree")}),
            flush = True)
    if ep:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
