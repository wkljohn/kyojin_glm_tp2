"""
N-gram prompt lookup ahead of a DFlash / DFlash2 drafter (GLM). Same rule as tools/qwen/spec_policy.install_lookup (which
wraps the MTP draft step): when the last >= lmin context tokens occurred before, the round verifies the K tokens that
followed them instead of the drafter's block, and skips the drafter forward (DFlash drafter K/V comes from target hidden
states, parked and flushed by the generator, so a skipped draft leaves no hole). A per-request gate compares the mean tokens
gained per lookup round with the drafter's and probes periodically. Everything is decided from token ids only, so expert-
parallel ranks take identical decisions.
"""
import os, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qwen"))
from spec_policy import SeqFollower   # noqa: E402


def install_dflash_lookup(g, ngram, stats = None):
    fol = SeqFollower(ngram[0]); K, lmin = ngram[1], ngram[2]; cmin = ngram[3] if len(ngram) > 3 else 0
    orig = type(g).iterate_draftmodel_dflash_gen.__get__(g)
    orig_enqueue = g.enqueue

    def enq(job):
        r = orig_enqueue(job)
        for j in (job if isinstance(job, list) else [job]):
            if len(getattr(j, "sequences", ())) == 1:
                fol.sync(j.sequences[0].sequence_ids)
        return r
    g.enqueue = enq

    gate = os.environ.get("QWSPEC_NGRAM_GATE", "1") != "0"
    PROBE, A = 32, 0.25
    dr0 = lambda: 1.0 + 0.6 * max(1, int(g.num_draft_tokens))
    st = {"sid": None, "n": 0, "kind": None, "ng": float(K + 1), "dr": dr0(), "skip": 0}

    def gate_ok(sid, n):
        if sid is not st["sid"]:
            st.update(sid = sid, kind = None, ng = float(K + 1), dr = dr0(), skip = 0)
        elif st["kind"] is not None and n > st["n"]:
            k = st["kind"]; st[k] = (1 - A) * st[k] + A * (n - st["n"])
        st["n"] = n
        return not gate or st["ng"] >= st["dr"] or st["skip"] >= PROBE

    def wrapped(results):
        cont = None; sid = None
        if len(g.active_jobs) == 1 and g.active_jobs[0].is_prefill_done() and len(g.active_jobs[0].sequences) == 1:
            sid = g.active_jobs[0].sequences[0].sequence_ids
            idx = fol.sync(sid)
            L, c = idx.match(K)
            if L >= lmin and c:
                cont = c
                if cmin > 0:
                    a = idx.witness_len(c)
                    if a < len(c): cont = c[:a] if a >= cmin else None
            if not gate_ok(sid, len(sid)):
                cont = None
        if cont is None:
            if sid is not None:
                st["kind"] = "dr"; st["skip"] += 1
            return orig(results)
        st["kind"] = "ng"; st["skip"] = 0
        g._draft_conf_round = None   # the verified tokens are the copy: not a label for the drafter's calibrator
        if stats is not None:
            stats["ng"] = stats.get("ng", 0) + 1; stats["ng_tokens"] = stats.get("ng_tokens", 0) + len(cont)
        return torch.tensor([cont[:K]], dtype = torch.long)

    g.iterate_draftmodel_dflash_gen = wrapped
    return stats
