# Handover: GLM-5.3-Flash EXL3 Ablit on two Strix Halo hosts (expert parallel, TP=2)

An OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming, tool calls) for
`GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit` (163.6 GiB), split across two 128 GB Ryzen AI Max+ 395 machines joined by
RoCE v2. The model is not in this package; download it on both hosts (below).

Source: fork `wkljohn/kyojin_glm_tp2`, branch `research/glm53-exl3-tp2-20261021`; the commit is in `source/COMMIT`.
Design and measurements: `doc/ep_tp2.md` in the source.

## Measured (reference pair, 2026-10-21, 8-family bench, 2048-token prompts, 256 tokens, DFlash2 + n-gram)

| | previous package (9a76bdd) | this package (`KYOJIN_EP_SHARE=0.53`, `KYOJIN_Q8_LINEAR=1`) |
|---|---|---|
| Prefill | 729 t/s | 740 t/s (806 without the drafter loaded) |
| Decode | 27.4-28.5 t/s mean | 33.4 t/s mean (22.6 prose .. 54.3 recall) |
| Verify round | 143 ms | 118.5 ms |
| Memory per host | ~32 GiB free at 128K context | ~4 GiB more free on rank 1, about the same on rank 0 |
| Startup | ~62 s load + ~20 s warm-up | + ~10 s Q8 conversion at load |

What changed (source commit in `source/COMMIT`):
- **Prefill two-batch overlap**: each prefill chunk runs as two halves interleaved at the all-reduces, so one half's
  network transfer overlaps the other half's compute (always on; `KYOJIN_EP_TBO=0` disables).
- **Weighted split** (`KYOJIN_EP_SHARE`): rank 0 takes that share of experts, heads, channels and vocabulary. Lets a
  host with more sustained compute do more of the work (see "Tuning the split").
- **Q8 linears** (`KYOJIN_Q8_LINEAR`): the BF16 attention / shared expert / dense / lm_head / drafter weights are
  converted at load to 8-bit with an fp16 scale per 64 rows (llama.cpp Q8_0 class). Decode reads half the bytes.
- **Settings check**: at start both ranks compare every setting that changes what they compute and refuse to start
  if one differs (`EP settings differ across hosts: ...`).

## What each host needs

All of these were true on the reference hosts; the package was only validated there.

- **Hardware**: Ryzen AI Max+ 395 (gfx1151), 128 GB. One RDMA NIC per host on a direct link (reference:
  Mellanox ConnectX-4 Lx, RoCE v2, 25 GbE, MTU 9000 on the netdev, active IB MTU 4096).
- **OS**: Ubuntu 26.04, kernel 7.0.0-31-generic, in-kernel amdgpu; `rdma-core` / `libibverbs` with `mlx5_ib` loaded.
- **Kernel command line** (lets the GPU address ~124 GiB of the unified memory; required, the model needs ~91 GiB per host):
  `amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856`
- **memlock unlimited** for the service user (`/etc/security/limits.d/`: `* soft memlock unlimited`, `* hard memlock unlimited`;
  the systemd unit sets `LimitMEMLOCK=infinity` itself).
- **Network**: a static IPv4 on each RoCE port in one subnet (reference: 192.168.99.1 rank 0, 192.168.99.2 rank 1).
  `ibv_devinfo` shows `PORT_ACTIVE`; `show_gids` (or `ibv_devinfo -v`) gives the RoCE v2 IPv4 GID index (reference: 3).
  TCP port 29531 and the gloo/NCCL bootstrap ports must be open between the two hosts on that link.
- **Disk**: ~166 GiB for the models + ~12 GiB for this install, on both hosts.
- `uv`, `zstd`, `python3` (for the scripts), `curl`.
- Nothing else may hold the GPU memory (one server per host).

## Install (both hosts)

```bash
# 1. models, same paths on both hosts (revisions pinned; ~166 GiB)
hf download Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit --revision 57edefd2f5d9b371c8345883304d5af68b52fa24 \
    --local-dir /srv/models/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit
hf download incoai/GLM-5.3-Flash-DFlash2 --revision bf582e4eacc1810f76656d1811693ff6c6737d2a \
    --local-dir /srv/models/GLM-5.3-Flash-DFlash2
python3 deploy/verify_models.py --models-dir /srv/models          # sizes; add --hash for sha256 (slow)

# 2. runtime + source (~10 GiB unpacked runtime; checks SHA256SUMS first)
sudo mkdir -p /opt/kyojin-glm-tp2 && sudo chown $USER /opt/kyojin-glm-tp2
./deploy/install.sh . /opt/kyojin-glm-tp2

# 3. GPU check
EXL3_ROCM_SDK=/opt/kyojin-glm-tp2/runtime/lib/python3.14/site-packages/_rocm_sdk_devel \
    bash /opt/kyojin-glm-tp2/kyojin_glm_tp2/tools/strix_halo/env.sh --check

# 4. per-host config + service
sudo cp deploy/node.env.example /etc/kyojin-ep.env      # edit: KYOJIN_EP_RANK, NCCL_IB_HCA, *_IFNAME, MASTER_ADDR, paths
sudo mkdir -p /var/log/kyojin-ep && sudo chown $USER /var/log/kyojin-ep
sed "s/REPLACE_WITH_SERVICE_USER/$USER/" deploy/kyojin-ep.service | sudo tee /etc/systemd/system/kyojin-ep.service
sudo systemctl daemon-reload && sudo systemctl enable kyojin-ep
```

The two hosts differ only in `KYOJIN_EP_RANK`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME`.
The source tree, runtime, models and every other setting must be identical: rank 1 replays rank 0's work and both
must compute the same thing.

## Start, check, stop

```bash
PEER=user@<rank-1 host> ./deploy/restart.sh      # from rank 0: stops both, starts rank 1 then rank 0
journalctl -fu kyojin-ep                          # ready at "ep_serve: listening on <host>:<port>"
./deploy/smoke_test.sh http://127.0.0.1:8083      # greedy, sampled, tool call, streaming -> PASS
```

Always start and stop the pair together. Rank 0 serves HTTP; rank 1 has no listener. Healthy logs show
`NET/IB : Using [0]mlx5_...` (RDMA, not sockets) and no `FAULT`.

## Operating notes

- **One request at a time**; others queue. The prompt cache (prefix reuse) is what makes agent follow-up turns fast.
- **No auth**: the server binds `HOST:PORT` unauthenticated. Put a reverse proxy with auth in front (reference: Caddy
  `reverse_proxy` to the backend with a `Bearer` header check, `flush_interval -1`, `response_header_timeout 900s`).
- **Fail-closed**: the two ranks must run the same work in the same order.
  - Before every decode/verify step they check that their rows agree. A mismatch logs `ep_serve: FAULT ...
    exiting (70)` on both and both processes exit; `Restart=on-failure` brings the pair back (~90 s).
  - A collective one rank never joins aborts the process after `KYOJIN_EP_NCCL_TIMEOUT` seconds (default 300)
    instead of hanging forever.
  - After every request, finished or cancelled, both ranks compare a digest of the generated tokens. A mismatch
    marks the pair faulted and every later request gets HTTP 503 until restart.
  - Report any `FAULT` line.
- **Accuracy (Q8)**: logits over the 8 bench texts compared with BF16, at 1K and 6K tokens. KL(BF16 || Q8) is
  0.139 / 0.338 nats per token overall, the same size as the difference between two BF16 builds that only sum in a
  different order (0.125 / 0.328); top-1 agreement 0.93; NLL within +-1 %. Two families (identifier-heavy dossier,
  long math) differ a lot between any two numeric variants, BF16 included (DSA top-k selection flips). Set
  `KYOJIN_Q8_LINEAR=0` for the unmodified BF16 weights.
- **Diagnostics** (env in `/etc/kyojin-ep.env`, off by default): `KYOJIN_EP_DRAFT_DEBUG=1` makes rank 1 log every
  round where its own drafter output differed from rank 0's (rank 0's is always used).
- **No image input** (HTTP 400). Vision is not loaded in the EP server.
- **Sampling** is supported (each request gets one seed shared by both ranks). Default sampling comes from the pack's
  `generation_config.json` (temperature 1.0, top_p 0.95) when a request omits it.
- **Model name**: requests must use `MODEL_ID`, or a name given with `--alias` in `EXTRA_ARGS`.
- **Logs**: `journalctl -u kyojin-ep`; per-request numbers in `KYOJIN_REQLOG_DIR`; counters at `GET /metrics`
  (Prometheus text, llama.cpp names).
- **Licenses**: the DFlash2 drafter is CC BY-NC-ND 4.0 (non-commercial). Check the model pack's own LICENSE.

## Tuning the split

`KYOJIN_EP_SHARE` should match the hosts' relative sustained GPU compute. On each host:

```bash
cd /opt/kyojin-glm-tp2/kyojin_glm_tp2 && EXL3_ROCM_SDK=/opt/kyojin-glm-tp2/runtime/lib/python3.14/site-packages/_rocm_sdk_devel \
    bash -c 'source tools/strix_halo/env.sh; .venv/bin/python tools/glm/gpu_speed_probe.py'   # service stopped
```

With GEMM throughputs T0 (rank 0) and T1 (rank 1): share = T0 / (T0 + T1), rounded to 0.50 / 0.53 / 0.56. Reference
pair: 36.3 and 30.3 TFLOPS gives 0.545; 0.53 measured best (0.50: decode 31.8, 0.56: decode 33.5 but no faster
rounds). Equal hosts: 0.5. The value must be the same in both `/etc/kyojin-ep.env` files.

## Updating

A new package differs from the old one in `source/` (and maybe `deploy/`); the runtime only changes if its checksum
does. Copy the new package over the old one with `rsync -a` (only the changed files move), then from rank 0:

```bash
PEER=user@<rank-1 host> ./deploy/update.sh . /opt/kyojin-glm-tp2   # stops both, syncs the package to the peer
                                                                   # (without runtime/), updates both, starts both
./deploy/smoke_test.sh http://127.0.0.1:8083
```

**From the 9a76bdd package**: add the two new settings to `/etc/kyojin-ep.env` on both hosts before running
update.sh (without them the new code runs like the old layout: share 0.5, BF16 weights):

```
KYOJIN_EP_SHARE=0.53      # or the value from "Tuning the split"
KYOJIN_Q8_LINEAR=1
```

The previous tree stays at `/opt/kyojin-glm-tp2/kyojin_glm_tp2.prev`. To roll back, stop both ranks, swap the two
directories on both hosts, and start them again.

## opencode

```json
"provider": {
  "glm-tp2": {
    "npm": "@ai-sdk/openai-compatible",
    "name": "GLM-5.3 Flash Ablit (2-node EXL3 TP2)",
    "options": { "baseURL": "https://<your proxy>/v1", "apiKey": "{env:GLM_TP2_API_KEY}" },
    "models": { "GLM-5.3-Flash-EXL3-Ablit-TP2": { "name": "GLM-5.3 Flash Ablit TP2", "tool_call": true, "reasoning": true,
                "limit": { "context": 131072, "output": 16384 } } }
  }
}
```

## Known limits

- Validated only on the reference pair (same kernel, ROCm runtime and NIC). The prebuilt extension targets gfx1151 and
  the shipped torch 2.13 / ROCm 10 runtime; rebuilding needs `tools/strix_halo/env.sh build` with that runtime's
  `_rocm_sdk_devel`.
- Accuracy is gated at the logit level (Q8 vs BF16, `tools/glm/ep_logits.py`), not against an unsplit single-host
  reference (the model does not fit one host); the two ranks agree bit-for-bit. Speculative decoding and any change of
  summation order can change greedy text at near-ties.
- A crash of either rank takes the pair down; `Restart=on-failure` restarts the crashed side, but if the pair does not
  re-form within a minute, run `restart.sh`.
