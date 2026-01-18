import os
import sys
import argparse
import time
from pathlib import Path
import torch
from accelerate.utils import set_seed
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pipeline.stream_diffvsr_pipeline import StreamDiffVSRPipeline, ControlNetModel, UNet2DConditionModel
from diffusers import DDIMScheduler
from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny


def get_device():
    """
    Detect and return the best available device.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # Check if MPS is actually usable (not just available)
        try:
            torch.zeros(1).to('mps')
            return torch.device('mps')
        except Exception:
            pass
    return torch.device('cpu')


def get_device_config(device):
    """
    Return device-specific configuration settings.
    """
    config = {
        'dtype': torch.float32,  # MPS works best with float32
        'enable_attention_slicing': False,
        'enable_vae_slicing': False,
        'enable_xformers': False,
    }

    if device.type == 'cuda':
        config['dtype'] = torch.float16
        config['enable_xformers'] = True
        # Enable TF32 for faster computation on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
    elif device.type == 'mps':
        config['dtype'] = torch.float32  # MPS doesn't fully support float16
        config['enable_attention_slicing'] = True  # Reduces memory on MPS
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
    parser = argparse.ArgumentParser(description="Test code for Stream-DiffVSR.")
    parser.add_argument("--model_id", default='stabilityai/stable-diffusion-x4-upscaler', type=str, help="model_id of the model to be tested.")
    parser.add_argument("--unet_pretrained_weight", type=str, help="UNet pretrained weight.")
    parser.add_argument("--controlnet_pretrained_weight", type=str, help="ControlNet pretrained weight.")
    parser.add_argument("--temporal_vae_pretrained_weight", type=str, help="Path to Temporal VAE.")
    parser.add_argument("--out_path", default='./StreamDiffVSR_results/', type=str, help="Path to output folder.")
    parser.add_argument("--in_path", type=str, required=True, help="Path to input folder (containing sets of LR images).")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of sampling steps")
    parser.add_argument("--enable_tensorrt", action='store_true', help="Enable TensorRT. Note that the performance will drop if TensorRT is enabled. (CUDA only)")
    parser.add_argument("--image_height", type=int, default=720, help="Height of the output images. Needed for TensorRT.")
    parser.add_argument("--image_width", type=int, default=1280, help="Width of the output images. Needed for TensorRT.")
    parser.add_argument("--device", type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help="Device to use for inference.")
    return parser.parse_args()


def load_component(cls, weight_path, model_id, subfolder):
    path = weight_path if weight_path else model_id
    sub = None if weight_path else subfolder
    return cls.from_pretrained(path, subfolder=sub)


def main():
    args = parse_args()

    print("Run with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    set_seed(42)

    # Device selection
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)

    device_config = get_device_config(device)
    print(f"\nUsing device: {device}")
    print(f"Device config: {device_config}")

    # TensorRT only works with CUDA
    if args.enable_tensorrt and device.type != 'cuda':
        print("Warning: TensorRT is only available on CUDA devices. Disabling TensorRT.")
        args.enable_tensorrt = False

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
            print("xFormers memory efficient attention enabled")
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

    seqs = sorted(os.listdir(args.in_path))
    total_frames = 0
    total_time = 0

    for seq in seqs:
        seq_path = os.path.join(args.in_path, seq)
        if not os.path.isdir(seq_path):
            continue

        frame_names = sorted(os.listdir(seq_path))
        frames = []
        for frame_name in frame_names:
            frame_path = os.path.join(seq_path, frame_name)
            if frame_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                with Image.open(frame_path) as im:
                    frames.append(im.convert("RGB").copy())

        if not frames:
            print(f"No valid frames found in {seq_path}, skipping...")
            continue

        print(f"\nProcessing {seq} ({len(frames)} frames)...")
        start_time = time.time()

        output = pipeline(
            '', frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=0,
            of_model=of_model
        )

        elapsed = time.time() - start_time
        total_frames += len(frames)
        total_time += elapsed

        frames_hr = output.images
        frames_to_save = [frame[0] for frame in frames_hr]

        seq_path_obj = Path(seq_path)
        target_path = os.path.join(args.out_path, seq_path_obj.parent.name, seq_path_obj.name)
        os.makedirs(target_path, exist_ok=True)

        for frame, name in zip(frames_to_save, frame_names):
            frame.save(os.path.join(target_path, name))

        fps = len(frames) / elapsed
        print(f"Upscaled {seq} and saved to {target_path}.")
        print(f"  Time: {elapsed:.2f}s | FPS: {fps:.2f} | Per frame: {elapsed/len(frames):.3f}s")

        del frames
        del frames_to_save
        clear_memory(device)

    if total_frames > 0:
        print(f"\n{'='*50}")
        print(f"Total: {total_frames} frames in {total_time:.2f}s")
        print(f"Average: {total_frames/total_time:.2f} FPS | {total_time/total_frames:.3f}s per frame")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
