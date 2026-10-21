#!/usr/bin/env bash
# Install the GLM-5.3 EP server from the handover package on ONE host (run it on both).
#   ./install.sh <package dir> [prefix=/opt/kyojin-glm-tp2]
# Creates <prefix>/runtime (python 3.14.3 venv + the shipped ROCm/torch site-packages) and <prefix>/kyojin_glm_tp2
# (source, prebuilt gfx1151 extension, .venv layered on the runtime). Needs uv (https://docs.astral.sh/uv/) and zstd.
set -euo pipefail
PKG="$(cd "$1" && pwd)"
PREFIX="${2:-/opt/kyojin-glm-tp2}"
PY=3.14.3

( cd "$PKG" && sha256sum --strict --check SHA256SUMS )

mkdir -p "$PREFIX"
command -v uv > /dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
uv python install "$PY"
if [ ! -x "$PREFIX/runtime/bin/python" ]; then
    uv venv --python "$PY" "$PREFIX/runtime"
fi
SP="$PREFIX/runtime/lib/python3.14/site-packages"
echo "[install] unpacking runtime site-packages (~10 GiB) into $SP"
zstd -dc "$PKG"/runtime/runtime-site-packages.tar.zst | tar -x -C "$SP"

SRC="$PREFIX/kyojin_glm_tp2"
mkdir -p "$SRC"
tar -xzf "$PKG"/source/kyojin_glm_tp2-*.tar.gz -C "$SRC"
cp "$PKG"/source/exllamav3_ext.cpython-314-x86_64-linux-gnu.so "$SRC/"
if [ ! -x "$SRC/.venv/bin/python" ]; then
    "$PREFIX/runtime/bin/python" -m venv --without-pip "$SRC/.venv"
fi
echo "$SP" > "$SRC/.venv/lib/python3.14/site-packages/kyojin_runtime.pth"

echo "[install] import check (no GPU work)"
export EXL3_ROCM_SDK="$SP/_rocm_sdk_devel"
cd "$SRC"
# shellcheck disable=SC1091
source tools/strix_halo/env.sh
.venv/bin/python -c "import torch, exllamav3_ext, exllamav3; import torch.distributed; print('torch', torch.__version__, 'hip', torch.version.hip, 'ext OK')"
echo "[install] done. GPU check: EXL3_ROCM_SDK=$EXL3_ROCM_SDK $SRC/tools/strix_halo/env.sh --check"
