#!/bin/bash
# Stream-DiffVSR RunPod Inference Script
# Runs inference and compresses output to video on the pod before download

set -e

# Configuration
INPUT_DIR="${1:-./input}"
OUTPUT_DIR="${2:-./output}"
HEIGHT="${3:-720}"
WIDTH="${4:-1280}"
FPS="${5:-30}"
CRF="${6:-18}"

echo "=== Stream-DiffVSR Inference ==="
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Resolution: ${WIDTH}x${HEIGHT}"

# Fix dependencies (only needed once, but idempotent)
pip install -q 'numpy<2' 'diffusers>=0.25,<0.30' 'transformers>=4.36,<4.40' 'huggingface-hub>=0.21,<1.0' 2>/dev/null || true

# Increase file limit for large videos
ulimit -n 65535

# Run inference
echo "Starting inference..."
PYTHONUNBUFFERED=1 python inference.py \
    --model_id Jamichsu/Stream-DiffVSR \
    --in_path "$INPUT_DIR" \
    --out_path "$OUTPUT_DIR" \
    --image_height "$HEIGHT" \
    --image_width "$WIDTH" \
    --device cuda

# Find the output subdirectory (mirrors input structure)
OUTPUT_FRAMES=$(find "$OUTPUT_DIR" -name "frame_0001.png" -type f | head -1 | xargs dirname)

if [ -z "$OUTPUT_FRAMES" ]; then
    echo "Error: No output frames found"
    exit 1
fi

VIDEO_NAME=$(basename "$OUTPUT_FRAMES")
OUTPUT_VIDEO="$OUTPUT_DIR/${VIDEO_NAME}_upscaled.mp4"

echo "Compressing frames to video..."
ffmpeg -y -framerate "$FPS" \
    -i "$OUTPUT_FRAMES/frame_%04d.png" \
    -c:v libx264 -crf "$CRF" -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUTPUT_VIDEO"

# Report sizes
FRAMES_SIZE=$(du -sh "$OUTPUT_FRAMES" | cut -f1)
VIDEO_SIZE=$(du -sh "$OUTPUT_VIDEO" | cut -f1)
echo ""
echo "=== Complete ==="
echo "Frames: $FRAMES_SIZE (can be deleted)"
echo "Video: $VIDEO_SIZE -> $OUTPUT_VIDEO"
echo ""
echo "Download with:"
echo "  scp -P \$PORT root@\$HOST:$OUTPUT_VIDEO ."
