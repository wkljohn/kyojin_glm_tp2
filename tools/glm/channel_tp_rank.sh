#!/usr/bin/env bash
# Run one independently loaded GLM channel-TP rank. Start both physical hosts.
set -euo pipefail
if (( $# < 3 )); then
    echo "Usage: $0 RANK MODEL OUTPUT_PREFIX [generation arguments...]" >&2
    exit 2
fi
tp_rank=$1
tp_model=$2
tp_output=$3
shift 3
case "$tp_rank" in
    0) tp_hca=mlx5_0; tp_if=ens1f0np0 ;;
    1) tp_hca=mlx5_1; tp_if=ens1f1np1 ;;
    *) echo "RANK must be 0 or 1" >&2; exit 2 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export EXL3_ROCM_SDK="${EXL3_ROCM_SDK:-/home/wkljohn/Desktop/cc/glm53-ciru-runtime-env/lib/python3.14/site-packages/_rocm_sdk_devel}"
source tools/strix_halo/env.sh
export RANK="$tp_rank" WORLD_SIZE=2
export MASTER_ADDR="${MASTER_ADDR:-192.168.99.1}" MASTER_PORT="${MASTER_PORT:-29957}"
export NCCL_NET=IB NCCL_IB_DISABLE=0
export NCCL_IB_HCA="${NCCL_IB_HCA:-$tp_hca}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=$tp_if}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$tp_if}"
export NCCL_IB_GID_INDEX=3 NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_MIN_NCHANNELS=2 NCCL_MAX_NCHANNELS=2
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export OMP_NUM_THREADS=1 EXL3_HAD_PF_FAST=0
exec "$EXL3_VENV/bin/python" tools/glm/channel_tp_generate.py \
    --model "$tp_model" --output "$tp_output" "$@"
