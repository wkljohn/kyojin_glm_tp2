#!/usr/bin/env bash
# Restart the EP pair: both ranks must start together. Run on rank 0's host; PEER is the rank-1 host (ssh, sudo rights).
#   PEER=user@192.168.99.2 ./restart.sh
set -euo pipefail
: "${PEER:?set PEER=user@rank1-host}"
sudo systemctl stop kyojin-ep; ssh "$PEER" sudo systemctl stop kyojin-ep
ssh "$PEER" sudo systemctl start kyojin-ep; sudo systemctl start kyojin-ep
echo "started; follow with: journalctl -fu kyojin-ep   (ready when 'ep_serve: listening on' appears, ~90 s)"
