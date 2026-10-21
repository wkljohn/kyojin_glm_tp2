#!/usr/bin/env bash
# Run one EP job on both hosts: syncs this checkout's sources to the peer, starts rank 1 there, rank 0 here.
#   tools/glm/ep_run2.sh <logdir> <script.py> [args...]
# Logs: <logdir>/rank0.log here, <logdir>/rank1.log copied back when both exit. Exit status = rank 0's.
set -uo pipefail
PEER="${EP_PEER:-wkljohn@192.168.99.2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="$1"; shift
mkdir -p "$LOG"
rsync -a --exclude build --exclude '.rebuild.*' --exclude '__pycache__' --exclude '.venv' "$ROOT/" "$PEER:$ROOT/" || exit 2
rsync -a "$ROOT"/exllamav3_ext*.so "$PEER:$ROOT/" || exit 2
q=$(printf ' %q' "$@")
# Forward the tuning environment so both ranks run the same configuration (a mismatch can desynchronize them)
envs=$(env | grep -E '^(KYOJIN_|EXL3_|NCCL_|EP_PEER_)' | grep -v '^NCCL_IB_HCA=\|^NCCL_SOCKET_IFNAME=' | while IFS= read -r kv; do printf ' %q' "$kv"; done)
ssh -n "$PEER" "cd $ROOT && { nohup env$envs tools/glm/ep_launch.sh$q > /tmp/ep_rank1.log 2>&1 < /dev/null & echo \$! > /tmp/ep_rank1.pid; } > /dev/null 2>&1"
"$ROOT/tools/glm/ep_launch.sh" "$@" > "$LOG/rank0.log" 2>&1 < /dev/null
rc=$?
for _ in $(seq 1 60); do ssh -n "$PEER" 'kill -0 $(cat /tmp/ep_rank1.pid) 2>/dev/null' || break; sleep 2; done
scp -q "$PEER:/tmp/ep_rank1.log" "$LOG/rank1.log"
echo "[ep_run2] rank0 exit $rc" >> "$LOG/rank0.log"
exit $rc
