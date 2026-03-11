#!/bin/bash
#
# 72B BitNet Training — Cloud (RunPod, Vast.ai, Lambda Labs)
#
# 1. Rent 8x A100 80GB instance
# 2. SSH or use web terminal
# 3. Run: bash run_72b_cloud.sh
#
# Set HF_TOKEN before running (for the-stack):
#   export HF_TOKEN=hf_xxxxxxxxxxxx
#

set -e

echo "=== Graviton-Native 72B Training ==="
echo ""

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Need NVIDIA GPUs."
    exit 1
fi
nvidia-smi
echo ""

# Find repo: current dir, or GRAVITON_REPO, or clone
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$REPO_DIR/pyproject.toml" ] && grep -q "graviton-native" "$REPO_DIR/pyproject.toml" 2>/dev/null; then
    echo "Using repo at $REPO_DIR"
elif [ -d "${GRAVITON_REPO:-}" ]; then
    REPO_DIR="$GRAVITON_REPO"
    echo "Using GRAVITON_REPO=$REPO_DIR"
else
    REPO_DIR="${GRAVITON_REPO:-/workspace/graviton-native}"
    if [ ! -d "$REPO_DIR" ]; then
        echo "Cloning graviton-native..."
        git clone https://github.com/opengraviton/graviton-native.git "$REPO_DIR" || {
            echo "Clone failed. Upload your repo and set GRAVITON_REPO=/path"
            exit 1
        }
    fi
fi
cd "$REPO_DIR"

# Install
echo "Installing dependencies..."
pip install -e ".[train]" -q
pip install deepspeed -q

# HF token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "WARNING: HF_TOKEN not set. the-stack requires it."
    echo "  export HF_TOKEN=hf_xxx"
    echo "  Or use --dataset hug-stack (smaller, no auth)"
    echo ""
fi

# Run
echo "Starting 72B training..."
graviton-train run --num_gpus 8 --model_size 72b --dataset the-stack \
  --steps 100000 --batch_size 1 --grad_accum 64 --seq_len 2048 \
  --output_dir /workspace/checkpoints --save_every 1000

echo ""
echo "Done. Checkpoints: /workspace/checkpoints/bitnet-72b/"
