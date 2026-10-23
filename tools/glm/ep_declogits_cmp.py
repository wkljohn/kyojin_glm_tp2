"""
Decode-path logit gate: compare two ep_bench --save-logits runs (same prompts, greedy). Decode positions are
compared only up to the first generated token where the runs differ (after that they condition on different text).

    python3 tools/glm/ep_declogits_cmp.py <ref run dir> <test run dir>

Prints per family the agreed prefix length, mean / max KL(ref || test) over those positions and the top-1
agreement, then the pooled mean. Judge against a noise floor run (ref vs a harmless reordering of ref).
"""
import os, sys
import torch


def main(ref_dir, test_dir):
    tot_kl = tot_n = 0.0
    for f in sorted(os.listdir(ref_dir)):
        if not (f.startswith("logits-") and f.endswith(".pt")) or not os.path.exists(os.path.join(test_dir, f)):
            continue
        r = torch.load(os.path.join(ref_dir, f))
        t = torch.load(os.path.join(test_dir, f))
        ir, it = r["ids"], t["ids"]
        n = min(len(ir), len(it), r["logits"].shape[0], t["logits"].shape[0])
        diff = (ir[:n] != it[:n]).nonzero()
        agree = int(diff[0]) if len(diff) else n
        # position i's logits produced token i; positions <= agree saw identical context
        m = min(agree + 1, n)
        lr = torch.log_softmax(r["logits"][:m].float(), -1)
        lt = torch.log_softmax(t["logits"][:m].float(), -1)
        pr = lr.exp()
        kl = torch.where(pr > 0, pr * (lr - lt), torch.zeros_like(pr)).sum(-1)   # -inf (masked vocab) terms count 0
        top1 = (lr.argmax(-1) == lt.argmax(-1)).float().mean()
        print(f"{f[7:-3]:8s} agreed {agree:4d}/{n}  KL mean {kl.mean():.5f}  max {kl.max():.4f}  top1 {top1:.4f}")
        tot_kl += float(kl.sum())
        tot_n += m
    print(f"{'all':8s} KL mean {tot_kl / max(tot_n, 1):.5f}  positions {int(tot_n)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
