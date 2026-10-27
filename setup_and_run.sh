#!/usr/bin/env bash
set -e

echo "===================================================="
echo " Starting 1-Click Kaggle TPU v5e-8 Deployment Setup "
echo "===================================================="

# 1. Use /tmp for large downloads & caches to avoid Kaggle's 20GB /kaggle/working limit
export HF_HOME="/tmp/huggingface"
export VLLM_CACHE_DIR="/tmp/vllm_cache"
export DATA_DIR="/tmp/open-webui-data"
mkdir -p "$HF_HOME" "$VLLM_CACHE_DIR" "$DATA_DIR"

# 2. Verify TPU v5e availability
echo "[1/5] Checking TPU v5e hardware..."
python3 -c "import torch_xla.core.xla_model as xm; print('TPU Devices found:', xm.get_xla_supported_devices())" || echo "Warning: Ensure Kaggle Accelerator is set to TPU VM v5e-8!"

# 3. Clone NVIDIA Nemotron repository as requested
echo "[2/5] Cloning NVIDIA-NeMo/Nemotron repository..."
if [ ! -d "/tmp/Nemotron" ]; then
    git clone --depth 1 https://github.com/NVIDIA-NeMo/Nemotron.git /tmp/Nemotron || true
fi

# 4. Install TPU Inference Engine, Open WebUI, and Cloudflared
echo "[3/5] Installing dependencies (HuggingFace Hub, Open WebUI, Cloudflared)..."
pip install -q -U huggingface_hub[cli] hf_transfer open-webui

# Install Cloudflared binary for zero-config public tunneling
if [ ! -f "/usr/local/bin/cloudflared" ]; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

# 5. Hand off to the Python Orchestrator to download weights, launch server, & open UI
echo "[4/5] Launching Python Orchestrator..."
python3 "$(dirname "$0")/orchestrator.py"