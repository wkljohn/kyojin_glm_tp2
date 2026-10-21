# GLM-5.3-Flash across two Strix Halo hosts (expert parallel, TP=2)

This fork runs GLM-5.3-Flash EXL3 packs that do not fit one 128 GB machine (for example
`GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit`, 163.6 GiB) on two machines joined by RDMA.

## Layout

| Part | Placement |
| --- | --- |
| Routed experts (288 per layer, top 8) | split by expert: rank 0 owns 0..143, rank 1 owns 144..287 |
| Shared expert | rank 0 only (counted once) |
| KDA, MLA/DSA attention, indexer, dense MLP (layers 0-2), router, mHC, embedding, head | replicated |
| Communication | one fp32 all-reduce of the MoE output per sparse layer (42 per token), RCCL over RoCE v2 |

Each rank permutes its router columns (and the score-correction bias) so that its own experts become local
slots 0..143. Routing is permutation-equivariant, so both ranks select the same experts and weights; the
peer's experts land on slots 144..287, which Kyojin's head-slice MoE kernels (the same contract as the CPU
expert split: `exl3_dec_moe`, BC bszN, gfx12 grouped, generic fused path) treat as exact zeros. The all-reduce
sums the two partials, so both ranks hold the same hidden state and stay in lockstep. Code:
`exllamav3/model/ep.py` and `BlockSparseMLP.ep_maybe_split`.

Memory per host: ~72.8 GiB experts + ~18 GiB replicated weights + cache.

## Running

Both hosts need the same checkout, the same model directory, and a ROCm torch with RCCL
(here the `glm53-ciru-runtime-env` venv, layered under `.venv` by a `.pth` file).

```bash
# on each host (rank from hostname; see the script for the NIC / GID mapping)
tools/glm/ep_launch.sh tools/glm/ep_selftest.py                      # transport check
tools/glm/ep_launch.sh tools/glm/ep_generate.py --model <pack> --max-new 128
```

`ep_generate.py` runs greedy generation on both ranks and exits non-zero if rank 1's text differs from
rank 0's. Environment: `KYOJIN_EP_WORLD=2 KYOJIN_EP_RANK=<r> MASTER_ADDR MASTER_PORT` plus the usual
`NCCL_*` (`NCCL_NET=IB`, so a broken verbs path fails instead of falling back to sockets).

## Limits of this first version

- The grouped WMMA prefill kernel only admits full expert layers, so prefill uses the slower split-capable
  paths. Extending its masking is the main prefill speed item.
- Attention/KDA are replicated (each host reads ~11 GiB of BF16 attention weights per token). Head-splitting
  them as DS4 and vLLM do would halve that.
- MTP drafting and the HTTP server are not yet EP-aware: use `--num-draft 0` paths only.
