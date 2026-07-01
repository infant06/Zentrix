#!/usr/bin/env bash
# Build ZenLLM (the `zenllm` CLI/server and/or the `zentrix` Python bridge).
#
# Usage:
#   CUDA_COMPUTE_CAP=86 ./scripts/build.sh [cli|python|all] [--cpu]
#
# Pins the CUDA compute capability so kernels compile for your GPU.
# Examples of CUDA_COMPUTE_CAP: 75 (Turing), 80/86 (Ampere), 89 (Ada), 90 (Hopper).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TARGET="${1:-cli}"
CPU=0
[[ "${2:-}" == "--cpu" || "${1:-}" == "--cpu" ]] && CPU=1
[[ "${1:-}" == "--cpu" ]] && TARGET="cli"

if [[ "$CPU" -eq 1 ]]; then
  FEATURES="code-execution"
else
  FEATURES="cuda,code-execution"
  echo "[build] CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-<auto-detect>}"
fi

if [[ "$TARGET" == "cli" || "$TARGET" == "all" ]]; then
  echo "[build] cargo build --release -p zenllm --features $FEATURES"
  cargo build --release -p zenllm --features "$FEATURES"
fi

if [[ "$TARGET" == "python" || "$TARGET" == "all" ]]; then
  echo "[build] maturin develop --release --features \"$FEATURES\" (run inside your venv)"
  ( cd crates/zen-python && maturin develop --release --features "$FEATURES" )
fi

echo "[build] done."
