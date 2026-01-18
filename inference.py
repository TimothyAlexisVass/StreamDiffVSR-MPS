import os
import sys
import argparse
import time
import subprocess
import shutil
import warnings
from pathlib import Path
import torch
from accelerate.utils import set_seed
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# Suppress safety checker warning
warnings.filterwarnings('ignore', message='.*safety checker.*')

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pipeline.stream_diffvsr_pipeline import StreamDiffVSRPipeline, ControlNetModel, UNet2DConditionModel
from diffusers import DDIMScheduler
from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny

# Resolution presets: output resolution -> input dimensions for 4x upscaling
# All dimensions must be divisible by 8 for RAFT optical flow
RESOLUTION_PRESETS = {
    720: {'input_height': 176, 'input_width': 320},    # 720p output (1280x704) - 176*4=704, 320*4=1280
    1080: {'input_height': 272, 'input_width': 480},   # 1080p output (1920x1088) - 272*4=1088, 480*4=1920
    1440: {'input_height': 360, 'input_width': 640},   # 1440p output (2560x1440) - 360*4=1440, 640*4=2560
}


def get_device_config(device):
    """
    Return device-specific configuration settings.
    """
    config = {
        'dtype': torch.float32,
        'enable_attention_slicing': False,
        'enable_vae_slicing': False,
        'enable_xformers': False,
    }

    if device.type == 'cuda':
        config['dtype'] = torch.float16
        config['enable_xformers'] = True
        torch.backends.cuda.matmul.allow_tf32 = True
    elif device.type == 'mps':
        config['dtype'] = torch.float32
        config['enable_attention_slicing'] = True
        config['enable_vae_slicing'] = True

    return config


def clear_memory(device):
    """
    Clear memory cache for the given device.
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Stream-DiffVSR: Video Super-Resolution")
    parser.add_argument("--input", type=str, required=True, help="Input video file (mp4, mov, avi, etc.)")
    parser.add_argument("--output_path", type=str, default='./output/', help="Output directory for upscaled video.")
    parser.add_argument("--model_id", default='Jamichsu/Stream-DiffVSR', type=str, help="Model ID from HuggingFace.")
    parser.add_argument("--unet_pretrained_weight", type=str, help="UNet pretrained weight.")
    parser.add_argument("--controlnet_pretrained_weight", type=str, help="ControlNet pretrained weight.")
    parser.add_argument("--temporal_vae_pretrained_weight", type=str, help="Path to Temporal VAE.")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of sampling steps")
    parser.add_argument("--device", type=str, default='mps', choices=['mps', 'cuda'], help="Device to use for inference.")
    
    # MPS-specific options
    parser.add_argument("--resolution", type=int, default=720, choices=[720, 1080, 1440],
                        help="Output resolution preset (720p, 1080p, 1440p). MPS only.")
    
    # TensorRT-specific options (CUDA only)
    parser.add_argument("--enable_tensorrt", action='store_true', help="Enable TensorRT acceleration. CUDA only.")
    parser.add_argument("--image_height", type=int, default=720, help="Output height for TensorRT. Requires --enable_tensorrt.")
    parser.add_argument("--image_width", type=int, default=1280, help="Output width for TensorRT. Requires --enable_tensorrt.")
    
    return parser.parse_args()


def validate_args(args):
    """Validate argument combinations and fail fast on invalid configs."""
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("Error: ffmpeg is not installed. Install it with: brew install ffmpeg")
    
    # Validate input file exists
    if not os.path.isfile(args.input):
        sys.exit(f"Error: Input file not found: {args.input}")
    
    # Validate input file is a video
    valid_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v')
    if not args.input.lower().endswith(valid_extensions):
        sys.exit(f"Error: Input must be a video file. Supported formats: {', '.join(valid_extensions)}")
    
    # --resolution is MPS only
    if args.device == 'cuda' and args.resolution != 720:
        sys.exit("Error: --resolution is only supported with --device mps. For CUDA, use --enable_tensorrt with --image_height and --image_width.")
    
    # --enable_tensorrt is CUDA only
    if args.enable_tensorrt and args.device != 'cuda':
        sys.exit("Error: --enable_tensorrt requires --device cuda.")
    
    # --image_height and --image_width require --enable_tensorrt
    # Check if user explicitly provided these args (not using defaults)
    if not args.enable_tensorrt:
        # Check if user explicitly set these (they differ from defaults)
        if args.image_height != 720 or args.image_width != 1280:
            sys.exit("Error: --image_height and --image_width require --enable_tensorrt.")


def extract_frames_from_video(video_path, output_dir):
    """
    Extract frames from video using ffmpeg.
    Returns the frame rate and number of frames extracted.
    """
    print(f"Extracting frames from video: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video FPS
    fps_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    
    try:
        fps_result = subprocess.run(fps_cmd, capture_output=True, text=True, check=True)
        fps_str = fps_result.stdout.strip()
        if '/' in fps_str:
            num, denom = map(float, fps_str.split('/'))
            fps = num / denom
        else:
            fps = float(fps_str)
    except subprocess.CalledProcessError:
        print("Warning: Could not detect FPS, defaulting to 30")
        fps = 30.0
    
    # Extract frames
    frame_pattern = os.path.join(output_dir, 'frame_%05d.png')
    extract_cmd = [
        'ffmpeg', '-i', video_path,
        '-qscale:v', '2',
        frame_pattern,
        '-y'
    ]
    
    try:
        subprocess.run(extract_cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error: Failed to extract frames from video. ffmpeg error: {e.stderr.decode()}")
    
    # Count frames
    frame_count = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"Extracted {frame_count} frames at {fps:.2f} FPS")
    
    return fps, frame_count


def encode_frames_to_video(frames_dir, output_path, fps):
    """
    Encode frames back to video using ffmpeg with original framerate.
    """
    print(f"\nEncoding {len(os.listdir(frames_dir))} frames to video at {fps:.2f} FPS...")
    frame_pattern = os.path.join(frames_dir, 'frame_%05d.png')
    
    encode_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    try:
        result = subprocess.run(encode_cmd, capture_output=True, check=True)
        print(f"✓ Video saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error: Failed to encode video. ffmpeg error: {e.stderr.decode()}")


def load_component(cls, weight_path, model_id, subfolder):
    path = weight_path if weight_path else model_id
    sub = None if weight_path else subfolder
    return cls.from_pretrained(path, subfolder=sub)


def main():
    args = parse_args()
    validate_args(args)

    # Suppress CUDA availability warning on MPS
    if args.device == 'mps':
        warnings.filterwarnings('ignore', message='.*CUDA is not available.*')

    device = torch.device(args.device)
    device_config = get_device_config(device)
    
    # Get input dimensions based on resolution preset (MPS) or TensorRT config (CUDA)
    if args.device == 'mps':
        preset = RESOLUTION_PRESETS[args.resolution]
        input_height = preset['input_height']
        input_width = preset['input_width']
        output_height = input_height * 4
        output_width = input_width * 4
    else:
        # CUDA path - dimensions come from TensorRT args if enabled
        input_height = None
        input_width = None
        if args.enable_tensorrt:
            output_height = args.image_height
            output_width = args.image_width
        else:
            output_height = None
            output_width = None

    print("="*60)
    print("Stream-DiffVSR: Video Super-Resolution")
    print("="*60)
    print(f"Input video: {args.input}")
    print(f"Output path: {args.output_path}")
    print(f"Device: {device}")
    print(f"Device config: {device_config}")
    if args.device == 'mps':
        print(f"Resolution: {args.resolution}p (input: {input_width}x{input_height} -> output: {output_width}x{output_height})")
    print("="*60)

    set_seed(42)
    
    # Setup temporary directory for frames
    video_name = Path(args.input).stem
    temp_dir = Path('pipeline/tmp')
    frames_input_dir = temp_dir / video_name / 'input'
    frames_output_dir = temp_dir / video_name / 'output'
    
    try:
        # Extract frames from video
        fps, frame_count = extract_frames_from_video(args.input, str(frames_input_dir))

        # Load models
        print("\nLoading models...")
        controlnet = load_component(ControlNetModel, args.controlnet_pretrained_weight, args.model_id, "controlnet")
        unet = load_component(UNet2DConditionModel, args.unet_pretrained_weight, args.model_id, "unet")
        vae = load_component(TemporalAutoencoderTiny, args.temporal_vae_pretrained_weight, args.model_id, "vae")
        scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

        tensorrt_kwargs = {
            "custom_pipeline": "/acceleration/tensorrt/sd_with_controlnet_ST",
            "image_height": args.image_height,
            "image_width": args.image_width,
        } if args.enable_tensorrt else {"custom_pipeline": None}

        pipeline = StreamDiffVSRPipeline.from_pretrained(
            args.model_id,
            controlnet=controlnet,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            **tensorrt_kwargs
        )

        if args.enable_tensorrt:
            pipeline.set_cached_folder("Jamichsu/Stream-DiffVSR")

        pipeline = pipeline.to(device)

        # Apply device-specific optimizations
        if device_config['enable_xformers']:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("xFormers enabled")
            except Exception as e:
                print(f"xFormers not available: {e}")

        if device_config['enable_attention_slicing']:
            pipeline.enable_attention_slicing("auto")
            print("Attention slicing enabled")

        if device_config['enable_vae_slicing']:
            pipeline.enable_vae_slicing()
            print("VAE slicing enabled")

        of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
        of_model.requires_grad_(False)

        # Load frames
        print(f"\nLoading {frame_count} frames...")
        frame_names = sorted([f for f in os.listdir(frames_input_dir) if f.endswith('.png')])
        frames = []
        for frame_name in frame_names:
            frame_path = os.path.join(frames_input_dir, frame_name)
            with Image.open(frame_path) as im:
                frames.append(im.convert("RGB").copy())

        print(f"Processing {len(frames)} frames...")
        print(f"  Original frame size: {frames[0].size[0]}x{frames[0].size[1]}")
        if args.device == 'mps':
            print(f"  Will process as: {input_width}x{input_height} -> output: {output_width}x{output_height}")
        
        start_time = time.time()

        # Pass height/width to pipeline so it preprocesses correctly
        pipeline_kwargs = {
            'num_inference_steps': args.num_inference_steps,
            'guidance_scale': 0,
            'of_model': of_model
        }
        if args.device == 'mps':
            pipeline_kwargs['height'] = input_height
            pipeline_kwargs['width'] = input_width

        output = pipeline('', frames, **pipeline_kwargs)

        elapsed = time.time() - start_time

        frames_hr = output.images
        frames_to_save = [frame[0] for frame in frames_hr]

        avg_fps = len(frames) / elapsed
        print(f"\nProcessing complete!")
        print(f"  Time: {elapsed:.2f}s | {avg_fps:.2f} FPS | {elapsed/len(frames):.3f}s per frame")

        # Save upscaled frames temporarily for ffmpeg
        os.makedirs(frames_output_dir, exist_ok=True)
        print(f"\nSaving {len(frames_to_save)} upscaled frames for encoding...")
        for i, frame in enumerate(frames_to_save):
            output_name = f"frame_{i+1:05d}.png"
            frame.save(os.path.join(frames_output_dir, output_name))

        # Encode to video with original framerate
        os.makedirs(args.output_path, exist_ok=True)
        output_video = os.path.join(args.output_path, f"{video_name}_upscaled.mp4")
        encode_frames_to_video(str(frames_output_dir), output_video, fps)

        print(f"\n{'='*60}")
        print(f"Complete! Output saved to: {output_video}")
        print(f"{'='*60}")

        del frames
        del frames_to_save
        clear_memory(device)
        
    finally:
        # Cleanup temporary directory
        if temp_dir.exists():
            print("\nCleaning up temporary files...")
            shutil.rmtree(temp_dir)
            print("Cleanup complete.")


if __name__ == "__main__":
    main()
