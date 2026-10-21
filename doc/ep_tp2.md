# GLM-5.3-Flash across two Strix Halo hosts (TP=2)

This fork runs GLM-5.3-Flash EXL3 packs that do not fit one 128 GB machine (for example
`GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit`, 163.6 GiB) on two Ryzen AI Max+ 395 machines joined by RDMA
(Mellanox ConnectX-4 Lx, RoCE v2, RCCL `NET/IB`).

## Layout

| Part | Placement |
| --- | --- |
| Routed experts (288 per layer, top 8) | split by expert: rank 0 owns 0..143, rank 1 owns 144..287 |
| KDA linear attention (34 layers, 64 heads) | split by heads, 32 per rank; o_proj by input rows |
| MLA/DSA attention (11 layers, 64 heads) | split by heads (q_b, kv_b, o_proj); latent cache, q/kv LoRA-A and DSA indexer replicated |
| Shared expert, dense MLP layers 0-2 | split by 128-aligned intermediate channels |
| lm_head | split by vocabulary, logits all-gathered in rank order |
| Router, mHC, norms, embedding | replicated |
| DFlash2 drafter | replicated, its MLPs split by intermediate channels |
| Communication | one fp32 all-reduce per attention block and per MLP/MoE block (90 per forward), bf16 on the wire for payloads >= 64 rows |

Each rank permutes its router columns (and the score-correction bias) so its own experts are local slots 0..143.
Routing is permutation-equivariant, so both ranks select the same experts and weights; peer experts land on slots
144..287, which every MoE kernel path treats as exact zeros (batch-1 `exl3_dec_moe`, the R-row union kernels, the
generic fused path and the grouped WMMA prefill kernel, which skips ids >= its pointer table before dereferencing).
All splits keep the original EXL3/BF16 tensors; nothing is re-quantized. Every kernel on the path is deterministic
and the collectives sum in a fixed order, so both ranks hold bit-identical state; `ep_bench.py` checks this on every
case. Code: `exllamav3/model/ep.py`, `BlockSparseMLP.ep_maybe_split`, `GatedDeltaNet(head_range=...)`,
`MLAttention(head_range=...)`, `ep_slice_gated_mlp`, `EPVocabHead`, `architecture/dflash2.py`, `tools/glm/ep_spec.py`.

Decode at batch <= 8 is bandwidth-bound on the BF16 weights; `hgemm`'s wide GEMV (`EXL3_HIP_WIDE_GEMV`, 16-byte
loads, fixed-order k-split) replaces rocBLAS tiles there.

Memory per host: ~86 GiB in use, ~38 GiB free at 8K context.

## Running

Both hosts need the same checkout, the same model directory, and a ROCm torch with RCCL (here the
`glm53-ciru-runtime-env` venv, layered under `.venv` by a `.pth` file).

```bash
# rank 1 is started over ssh by the runner; rank from hostname, NIC/GID mapping in ep_launch.sh
tools/glm/ep_run2.sh <logdir> tools/glm/ep_selftest.py                       # transport check
tools/glm/ep_run2.sh <logdir> tools/glm/ep_generate.py --model <pack>        # lockstep greedy generation
tools/glm/ep_run2.sh <logdir> tools/glm/ep_bench.py --model <pack> --out <dir> \
    --draft <GLM-5.3-Flash-DFlash2> --num-draft 7 --dynamic-draft --draft-confidence 0.4 --ngram 2,7,16,3
```

`ep_bench.py` is modelled on ds4-bench: fixed 2048-token prompts (exact token count) from eight families
(technical dossier, code review, Italian prose, multi-turn chat, math, JSON aggregation, Chinese log, long-document
needle), a fixed number of greedy tokens, prefill and decode measured separately, speculative acceptance, and a
rank-agreement check of the generated ids. The EP tools apply the GLM server's tuned `SPEED_ENV` route table.
Switches: `KYOJIN_EP_{KDA,MLA,DENSE,SHARED,HEAD,DRAFT}_SPLIT=0`, `KYOJIN_EP_BF16_MIN=0`.

## Measured (2026-10-21, 2048-token prompts, 256 tokens, greedy)

| Configuration | Prefill t/s | Decode t/s |
| --- | --- | --- |
| first working EP build (generic MoE path, no drafter) | 49 | 7.0 |
| final, no drafter (single row) | ~730 | 14.0 |
| final, DFlash2 + dynamic draft + n-gram lookup | **727.5** | **28.4** mean (17.8 prose .. 47.1 needle) |

## Limits

- The HTTP server (`tools/glm/serve.py`) is not EP-aware yet: rank 1 has to mirror every generator call.
- MTP (layer 45) under EP is untested.
- RCCL calls at 8 verify rows are ~0.1 ms median but wait for the slower rank (expert load imbalance); a
  verbs-level all-reduce and frequency-based expert placement are the next items.
- The DFlash2 weights are CC BY-NC-ND (non-commercial).
