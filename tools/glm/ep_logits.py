"""
Logit-level accuracy gate for numeric changes (e.g. KYOJIN_Q8_LINEAR) under EP: one plain forward per bench family
(fixed-length real text, no cache), the full-vocabulary logits of the last --keep positions saved by rank 0.

    tools/glm/ep_run2.sh <log> tools/glm/ep_logits.py --model <pack> --out <dir>/<label>
    python3 tools/glm/ep_logits.py --compare <dir>/<ref label> <dir>/<test label>

--compare prints, per family and overall, KL(ref || test) in nats/token, top-1 agreement, and the next-token NLL of
both runs on the text itself.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch


def compare(ref_dir, test_dir):
    tot = [0.0, 0.0, 0.0, 0.0, 0]
    for f in sorted(os.listdir(test_dir)):
        if not f.endswith(".pt"):
            continue
        r = torch.load(os.path.join(ref_dir, f))
        t = torch.load(os.path.join(test_dir, f))
        lr = torch.log_softmax(r["logits"].float(), -1)
        lt = torch.log_softmax(t["logits"].float(), -1)
        kl = (lr.exp() * (lr - lt)).sum(-1)
        top1 = (lr.argmax(-1) == lt.argmax(-1)).float()
        nxt = r["next"]
        nll_r = -lr.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
        nll_t = -lt.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
        n = kl.numel()
        print(f"{f[:-3]:8s} KL {kl.mean():.5f}  max {kl.max():.4f}  top1 {top1.mean():.4f}  "
              f"NLL ref {nll_r.mean():.4f} test {nll_t.mean():.4f} (d {nll_t.mean() - nll_r.mean():+.4f})")
        for i, v in enumerate((kl.sum(), top1.sum(), nll_r.sum(), nll_t.sum())):
            tot[i] += float(v)
        tot[4] += n
    n = tot[4]
    print(f"{'all':8s} KL {tot[0] / n:.5f}  top1 {tot[1] / n:.4f}  NLL ref {tot[2] / n:.4f} test {tot[3] / n:.4f} "
          f"(d {(tot[3] - tot[2]) / n:+.4f}, {(tot[3] - tot[2]) / tot[2] * 100:+.2f} %)  positions {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--tokens", type = int, default = 1024)
    ap.add_argument("--keep", type = int, default = 512)
    ap.add_argument("--compare", nargs = 2)
    ap.add_argument("--families", default = None)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return

    from ep_bench import FAMILIES, build_ids    # first: importing it applies the GLM SPEED_ENV
    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.model.ep import ep_state
    ep = ep_state()
    rank = ep.rank if ep else 0
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    model.load(device = "cuda:0", progressbar = False)
    if rank == 0:
        os.makedirs(args.out, exist_ok = True)
    with torch.inference_mode():
        for fam in (args.families.split(",") if args.families else FAMILIES):
            ids = build_ids(tokenizer, fam, args.tokens + 1)
            x, nxt = ids[:, :-1], ids[0, -args.keep:]
            # head only for the kept positions (full-vocabulary logits of a 16K+ prompt would not fit)
            logits = model.forward(x, {"attn_mode": "flash_attn_nc", "last_tokens_only": args.keep})
            if rank == 0:
                torch.save({"logits": logits[0, -args.keep:].half().cpu(), "next": nxt.cpu()},
                           os.path.join(args.out, f"{fam}.pt"))
                print(f"[logits] {fam} saved", flush = True)
    if ep:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
