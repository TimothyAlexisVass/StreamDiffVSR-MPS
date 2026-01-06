#!/bin/bash
# Local script to upscale video using RunPod
# Usage: ./local_upscale.sh input_video.mp4 [height] [width]

set -e

INPUT_VIDEO="$1"
HEIGHT="${2:-720}"
WIDTH="${3:-1280}"
FPS="${4:-30}"

if [ -z "$INPUT_VIDEO" ]; then
    echo "Usage: $0 input_video.mp4 [height] [width] [fps]"
    exit 1
fi

VIDEO_NAME=$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')
WORK_DIR="$(pwd)/runpod_work"
FRAMES_DIR="$WORK_DIR/input/$VIDEO_NAME"

echo "=== Stream-DiffVSR Upscaler ==="
echo "Input: $INPUT_VIDEO"
echo "Output resolution: ${WIDTH}x${HEIGHT}"

# Extract frames locally
mkdir -p "$FRAMES_DIR"
echo "Extracting frames..."
ffmpeg -i "$INPUT_VIDEO" -qscale:v 2 "$FRAMES_DIR/frame_%04d.png" -y 2>/dev/null

FRAME_COUNT=$(ls "$FRAMES_DIR" | wc -l | tr -d ' ')
echo "Extracted $FRAME_COUNT frames"

# Get RunPod pod info
echo ""
echo "Enter RunPod SSH details:"
read -p "Host (e.g., 194.68.245.198): " RUNPOD_HOST
read -p "Port (e.g., 22030): " RUNPOD_PORT

SSH_CMD="ssh -p $RUNPOD_PORT root@$RUNPOD_HOST"
SCP_CMD="scp -P $RUNPOD_PORT"

# Upload frames
echo "Uploading frames to RunPod..."
rsync -avz --progress -e "ssh -p $RUNPOD_PORT" \
    "$FRAMES_DIR/" "root@$RUNPOD_HOST:/workspace/stream-diffvsr/input/$VIDEO_NAME/"

# Run inference on pod
echo "Running inference on RunPod..."
$SSH_CMD "cd /workspace/stream-diffvsr && bash scripts/runpod_inference.sh ./input ./output $HEIGHT $WIDTH $FPS"

# Download result
OUTPUT_VIDEO="${VIDEO_NAME}_upscaled.mp4"
echo "Downloading result..."
$SCP_CMD "root@$RUNPOD_HOST:/workspace/stream-diffvsr/output/${OUTPUT_VIDEO}" "./"

echo ""
echo "=== Complete ==="
echo "Output: ./$OUTPUT_VIDEO"

# Cleanup
rm -rf "$WORK_DIR"
