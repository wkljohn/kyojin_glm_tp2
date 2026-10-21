#!/usr/bin/env bash
# Update installed hosts to the source in this package. The runtime is kept (the package may omit runtime/); the
# extension .so is replaced from source/. The previous tree stays at <prefix>/kyojin_glm_tp2.prev for rollback.
#   PEER=user@rank1-host ./update.sh <package dir> [prefix=/opt/kyojin-glm-tp2]
#       run on rank 0: stops both ranks, copies the package to the peer (same path, without runtime/), updates both,
#       starts rank 1 then rank 0
#   ./update.sh <package dir> [prefix]
#       this host only; the service must be stopped (a running server would import a mix of old and new files)
set -euo pipefail
PKG="$(cd "$1" && pwd)"
PREFIX="${2:-/opt/kyojin-glm-tp2}"
HERE="$PKG/deploy/update.sh"

if [ -n "${PEER:-}" ]; then
    sudo systemctl stop kyojin-ep; ssh "$PEER" sudo systemctl stop kyojin-ep
    ssh "$PEER" mkdir -p "$PKG"
    rsync -a --delete --exclude runtime/ "$PKG"/ "$PEER:$PKG"/
    ssh "$PEER" "$HERE" "$PKG" "$PREFIX"
    "$HERE" "$PKG" "$PREFIX"
    ssh "$PEER" sudo systemctl start kyojin-ep; sudo systemctl start kyojin-ep
    echo "both hosts on $(cut -d' ' -f1 "$PKG/source/COMMIT"); follow with: journalctl -fu kyojin-ep (ready ~90 s)"
    exit 0
fi

if systemctl is-active --quiet kyojin-ep 2> /dev/null; then
    echo "kyojin-ep is running on $(hostname): stop the pair first, or run from rank 0 with PEER=..." >&2; exit 1
fi
( cd "$PKG" && sha256sum --strict --check --ignore-missing --quiet SHA256SUMS )
SP="$PREFIX/runtime/lib/python3.14/site-packages"
[ -d "$SP/torch" ] || { echo "no runtime at $PREFIX/runtime: run install.sh first" >&2; exit 1; }

SRC="$PREFIX/kyojin_glm_tp2"
NEW="$SRC.new"
rm -rf "$NEW"; mkdir -p "$NEW"
tar -xzf "$PKG/source/kyojin_glm_tp2-$(cut -d' ' -f1 "$PKG/source/COMMIT").tar.gz" -C "$NEW"
cp "$PKG"/source/exllamav3_ext.cpython-314-x86_64-linux-gnu.so "$NEW/"
"$PREFIX/runtime/bin/python" -m venv --without-pip "$NEW/.venv"
echo "$SP" > "$NEW/.venv/lib/python3.14/site-packages/kyojin_runtime.pth"

echo "[update] import check (no GPU work)"
(
    export EXL3_ROCM_SDK="$SP/_rocm_sdk_devel"
    cd "$NEW"
    # shellcheck disable=SC1091
    source tools/strix_halo/env.sh
    .venv/bin/python -c "import torch, exllamav3_ext, exllamav3; from exllamav3.model.ep import EPDesync; print('ext OK')"
)
rm -rf "$SRC.prev"
[ -d "$SRC" ] && mv "$SRC" "$SRC.prev"
mv "$NEW" "$SRC"
echo "[update] $(hostname): $(cut -d' ' -f1 "$PKG/source/COMMIT") installed; previous tree in $SRC.prev"
