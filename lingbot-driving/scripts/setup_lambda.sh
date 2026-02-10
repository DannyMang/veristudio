#!/bin/bash
# setup_lambda.sh — One-shot setup for Lambda GPU instance
# Run from: ~/bakadddd/veristudio/
set -e

echo "=== LingBot-World Lambda Setup ==="

# 0. Check GPU
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 1. Init submodule (lingbot-world)
echo "[2/5] Initializing lingbot-world submodule..."
git submodule update --init --recursive
ls lingbot-world/wan/ && echo "  -> lingbot-world OK" || echo "  -> FAILED: lingbot-world not populated"

# 2. Install dependencies
echo "[3/5] Installing Python dependencies..."
pip install -e lingbot-world/
pip install Pillow imageio[ffmpeg] imageio-ffmpeg

# 3. Download model weights (~28GB)
echo "[4/5] Downloading model weights (this takes a while)..."
pip install huggingface-hub[cli]
huggingface-cli download robbyant/lingbot-world-base-cam \
    --local-dir ./lingbot-world-base-cam

echo "[5/5] Checking weights..."
ls -lh lingbot-world-base-cam/ 2>/dev/null || echo "  -> WARNING: weights not found"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Quick test (no camera poses, uses built-in example):"
echo "  cd lingbot-world && python generate.py --task i2v-A14B --size 480*832 --ckpt_dir ../lingbot-world-base-cam --t5_cpu --frame_num 17 --sample_steps 30"
echo ""
echo "With your dashcam image + scripted route:"
echo "  python lingbot-driving/scripts/generate_demo.py --ckpt_dir ./lingbot-world-base-cam --init_image dashcam.jpg"
