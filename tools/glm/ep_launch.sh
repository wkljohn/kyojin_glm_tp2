#!/usr/bin/env bash
# One expert-parallel rank of GLM-5.3-Flash EXL3 on this host (two Strix Halo hosts over Mellanox RoCE v2).
# Rank comes from the hostname; run the same command on both hosts (rank 0 first or second, either works).
#
#   tools/glm/ep_launch.sh tools/glm/ep_generate.py --model <dir> [args...]
#   tools/glm/ep_launch.sh tools/glm/serve.py --model <dir> --num-draft 0 ...
#
# Transport is fail-closed RCCL over IB verbs (NCCL_NET=IB): if the verbs path is unusable RCCL errors out instead of
# silently using sockets. Override any variable from the environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CIRU_ENV="${CIRU_ENV:-/home/wkljohn/Desktop/cc/glm53-ciru-runtime-env}"

case "$(hostname)" in
    wkljohn-NucBox-EVO-X2)   RANK_DEFAULT=0 HCA_DEFAULT=mlx5_0 IF_DEFAULT=ens1f0np0 ;;
    wkljohn-NucBox-EVO-X2-2) RANK_DEFAULT=1 HCA_DEFAULT=mlx5_1 IF_DEFAULT=ens1f1np1 ;;
    *) echo "[ep_launch] unknown host $(hostname); set KYOJIN_EP_RANK, NCCL_IB_HCA, NCCL_SOCKET_IFNAME" >&2
       RANK_DEFAULT="" HCA_DEFAULT="" IF_DEFAULT="" ;;
esac

export KYOJIN_EP_WORLD="${KYOJIN_EP_WORLD:-2}"
export KYOJIN_EP_RANK="${KYOJIN_EP_RANK:-$RANK_DEFAULT}"
export MASTER_ADDR="${MASTER_ADDR:-192.168.99.1}"
export MASTER_PORT="${MASTER_PORT:-29531}"

export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="${NCCL_IB_HCA:-$HCA_DEFAULT}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-2}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-2}"
# Decode block graphs replay captured RCCL collectives between eager ones on the same communicator; RCCL deadlocks on
# that mix unless graph mixing support is on (tools/glm/ep_graph_probe.py)
export NCCL_GRAPH_MIXING_SUPPORT="${NCCL_GRAPH_MIXING_SUPPORT:-1}"
# Connect every channel at communicator init: with two communicators per process (torch + pynccl), no lazy connection
# setup can then interleave with the other communicator's collectives
export NCCL_RUNTIME_CONNECT="${NCCL_RUNTIME_CONNECT:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=$IF_DEFAULT}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IF_DEFAULT}"
# A collective one rank never joins aborts the process after KYOJIN_EP_NCCL_TIMEOUT (300 s) instead of hanging
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-3}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Kyojin runtime environment (HSA preload, PYTHONPATH); the SDK path is build-only (env.sh Trap 5)
export EXL3_ROCM_SDK="${EXL3_ROCM_SDK:-$CIRU_ENV/lib/python3.14/site-packages/_rocm_sdk_devel}"
cd "$ROOT"
# shellcheck disable=SC1091
source tools/strix_halo/env.sh

ulimit -l unlimited 2>/dev/null || true
echo "[ep_launch] rank $KYOJIN_EP_RANK/$KYOJIN_EP_WORLD hca $NCCL_IB_HCA gid $NCCL_IB_GID_INDEX master $MASTER_ADDR:$MASTER_PORT" >&2
# KYOJIN_EP_WRAP (forwarded to the peer) applies only on rank KYOJIN_EP_WRAP_RANK (default 1)
if [ -n "${KYOJIN_EP_WRAP:-}" ] && [ "${KYOJIN_EP_WRAP_RANK:-1}" = "$KYOJIN_EP_RANK" ]; then EP_WRAP="$KYOJIN_EP_WRAP"; fi
# EP_WRAP: optional profiler prefix, e.g. "rocprofv3 --kernel-trace --output-format csv -d DIR --"
exec ${EP_WRAP:-} "$ROOT/.venv/bin/python" "$@"
