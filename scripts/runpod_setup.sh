#!/bin/bash
# One-time RunPod setup script
# Run this after creating a new pod to install dependencies

set -e

echo "=== Setting up Stream-DiffVSR on RunPod ==="

cd /workspace

# Clone repo if not present
if [ ! -d "stream-diffvsr" ]; then
    git clone https://github.com/199-biotechnologies/stream-diffvsr.git
fi

cd stream-diffvsr

# Install Python dependencies
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q -r requirements.txt

# Fix known compatibility issues
pip install -q 'numpy<2'
pip install -q 'diffusers>=0.25,<0.30'
pip install -q 'transformers>=4.36,<4.40'
pip install -q 'huggingface-hub>=0.21,<1.0'

# Install ffmpeg if not present
which ffmpeg || apt-get update && apt-get install -y ffmpeg

echo ""
echo "=== Setup Complete ==="
echo "Upload your input frames to: /workspace/stream-diffvsr/input/<video_name>/"
echo "Then run: bash scripts/runpod_inference.sh"
