# GLM-5.3 Flash EXL3 on two physical gfx1151 hosts

The experimental channel-TP path runs one process on each host's local GPU 0.
Every routed expert is present on both hosts, with gate/up output channels
and down input channels split into complementary 1024-channel halves.
The original EXL3 trellis and sign vectors are sliced before GPU upload.
The shared expert runs on global rank 0. One FP32 sum per MoE layer combines
both channel contributions and the shared expert. Attention, embeddings,
and the output head are replicated.

This path supports the selected
`GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit` pack with unfused EXL3 routed
projections. It rejects CPU expert offload, dense routed projections, and
expert post normalization. Expanded-weight caching is disabled. The CLI
performs ordinary greedy text generation with the model's chat template.
It checks identical rank inputs and schedules, finite logits, agreement on
each greedy token, and deterministic results for repeated prompts.

## Build and runtime

Both hosts need identical source, the built `exllamav3_ext` extension, the
complete original model directory, and compatible ROCm/PyTorch runtimes.
Build with the existing Strix Halo instructions in `doc/install.md` and
`tools/strix_halo/env.sh`. The tested runtime is Python 3.14.3,
PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333, gfx1151. The extension was built
with `EXL3_HIP_DEFINES=EXL3_HIP_STG_PAD`.

`EXL3_VENV` selects the runtime environment; `EXL3_ROCM_SDK` selects the
matching SDK for HSA and runtime HIP kernel compilation. The rank wrapper's
default SDK path uses this workspace's existing CIRU runtime environment.
Run with the GPUs available for this trial; each rank holds roughly half
the routed expert payload plus the replicated modules and its cache.

## Two-host launch

From the checkout on **each** host, set absolute model and evidence paths:

```bash
export EXL3_VENV="$PWD/.venv"
export DS4_RESEARCH_ROOT=/home/wkljohn/Desktop/cc/research-results
MODEL=/home/wkljohn/Desktop/cc/models/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit
OUT="$DS4_RESEARCH_ROOT/kyojin-glm-tp2-20261021/manual-generation"
```

On coordinator `192.168.99.1`:

```bash
bash tools/glm/channel_tp_rank.sh 0 "$MODEL" "$OUT" \
  --context 512 --prefill-chunk 16 --max-new-tokens 32 --no-thinking \
  --prompt 'Reply with only the number: What is 2 + 2?' \
  --prompt 'Name the capital of France in one word.' \
  --prompt 'Reply with only the number: What is 2 + 2?'
```

On peer `192.168.99.2`, run the same arguments with rank `1`.
Start both commands within the process-group timeout. The wrapper selects
`mlx5_0` / `ens1f0np0` on rank 0 and `mlx5_1` / `ens1f1np1` on rank 1,
RoCE v2 GID index 3, two RCCL channels, and mandatory IB transport.
`MASTER_ADDR` and `MASTER_PORT` override the rendezvous address and port;
NIC and HCA variables can be set explicitly for another installation.

Capture both stdout/stderr streams. A completed check requires both processes
to exit 0, both `<OUT>.rank{0,1}.json` reports to have `complete: true`, and
the RCCL logs to show IB payload channels. Reports include prompts, token
IDs, decoded output, repeated-request agreement, effective environment,
runtime versions, and configuration/template/extension hashes. Record the
source commit and original weight hashes alongside them.

## Component check

```bash
source tools/strix_halo/env.sh
"$EXL3_VENV/bin/python" tools/glm/check_tp_channel_layer.py \
  --model "$MODEL" --output "$DS4_RESEARCH_ROOT/kyojin-glm-tp2-20261021/channel-layer3.json"
```

This compares the summed shards against the unsplit real-weight MoE layer,
checks exact packed reconstruction for all 864 routed projections, and
numerically exercises all 288 expert IDs. The separate
`tp2_collective_probe.py` checks real two-host transport and dtype preservation.
Neither component check alone establishes complete-model inference or a
performance result. This research branch is separate from DS4 performance
promotion and from the concurrent expert-parallel implementation.
