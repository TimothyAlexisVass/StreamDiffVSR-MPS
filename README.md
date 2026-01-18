# Stream-DiffVSR MPS: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion, Optimized for Metal Performance Shaders (MPS)
**Video Super-Resolution using Diffusion Models**

Based on [Stream-DiffVSR](https://github.com/jamichss/Stream-DiffVSR) by Shiu et al., with major fixes for memory efficiency and usability.<br>
**Original Authors:** Hau-Shiang Shiu, Chin-Yang Lin, Zhixiang Wang, Chi-Wei Hsiao, Po-Fan Yu, Yu-Chih Chen, Yu-Lun Liu


<a href='https://jamichss.github.io/stream-diffvsr-project-page/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://huggingface.co/Jamichsu/Stream-DiffVSR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20(v1)-blue"></a> &nbsp;
<a href="https://arxiv.org/abs/2512.23709"><img src="https://img.shields.io/badge/arXiv-2510.12747-b31b1b.svg"></a>

---

## What optimization was done for MPS?

The [original implementation](https://github.com/jamichss/Stream-DiffVSR) loads **all frames into memory** before processing (all upscaled images, all optical flows, all latent tensors). This causes memory inefficiencies on a reasonably long video. Only 2 frames were processed at most, so now only 2 frames will be loaded into memory at a time.
This project also uses the improvements from [Boris Djordjevic's project](https://github.com/199-biotechnologies/stream-diffvsr)

### Key Improvements
- **Frame-by-frame processing** - Only current frame + previous output in memory at any time
- **On-demand optical flow** - Computed per frame pair, not pre-computed for entire video
- **Direct video input** - Set almost any video format with `--input path/to/file.mp4`
- **Attention slicing** - Reduces memory usage on unified memory
- **VAE slicing** - Processes VAE in chunks for large images
- **Device-agnostic code** - All hardcoded CUDA references removed
- **Streamlined dependencies** - Removed 50+ NVIDIA-specific packages

---

## Abstract

Diffusion-based video super-resolution (VSR) methods achieve strong perceptual quality but remain impractical for latency-sensitive settings due to reliance on future frames and expensive multi-step denoising. We propose Stream-DiffVSR, a causally conditioned diffusion framework for efficient online VSR. Operating strictly on past frames, it combines a four-step distilled denoiser for fast inference, an Auto-regressive Temporal Guidance (ARTG) module injecting motion-aligned cues during latent denoising, and a lightweight temporal-aware decoder with a Temporal Processor Module (TPM) enhancing detail and temporal coherence.

---

## Installation

### Mac / Apple Silicon (Recommended)

```bash
git clone git@github.com:TimothyAlexisVass/StreamDiffVSR-MPS.git
cd StreamDiffVSR-MPS

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

**Requirements:**
- macOS 12.3+ (Monterey or later)
- Python 3.9+
- Apple Silicon Mac (M1/M2/M3/M4)
- 8GB minimum (16GB+ RAM recommended)
- ffmpeg (for video input)

### Linux / NVIDIA GPU

```bash
git clone git@github.com:TimothyAlexisVass/StreamDiffVSR-MPS.git
cd StreamDiffVSR-MPS

conda env create -f requirements.yml
conda activate stream-diffvsr
```

---

## Usage

### Basic Inference

#### MPS
```bash
python inference.py --input path/to/video.mp4 --device mps
```

#### CUDA
```bash
python inference.py --input path/to/video.mp4 --device cuda
```

The script automatically:
- Detects the best device (MPS on Mac, CUDA on NVIDIA)
- Processes one frame at a time (memory efficient)
- Uses full-resolution optical flow for temporal consistency
- Extracts frames from video files automatically

### All Options

```bash
python inference.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | *required* | Input video file (mp4, mov, avi, mkv, etc.) |
| `--output_path` | `./output/` | Output directory for upscaled video |
| `--device` | `mps` | `mps` or `cuda` |
| `--output_resolution` | `720` | Output resolution preset: `720`, `1080`, or `1440` (MPS only) |
| `--num_inference_steps` | `4` | Denoising steps (more = slower, higher quality) |
| `--model_id` | `Jamichsu/Stream-DiffVSR` | HuggingFace model ID |
| `--unet` | Stream-DiffVSR UNet | Custom UNet pretrained weight path (optional) |
| `--controlnet` | Stream-DiffVSR ControlNet | Custom ControlNet pretrained weight path (optional) |
| `--temporal_vae` | Stream-DiffVSR VAE | Custom Temporal VAE weight path (optional) |
| `--enable_tensorrt` | off | Enable TensorRT acceleration (CUDA only) |
| `--image_height` | `720` | Output height for TensorRT (requires `--enable_tensorrt`) |
| `--image_width` | `1280` | Output width for TensorRT (requires `--enable_tensorrt`) |

---

## Approximate Performance

| Device | Chip | RAM | 720p Frame Time | Notes |
|--------|------|-----|-----------------|-------|
| Mac | M1 | 8GB | ~4-6s | May need lower resolution |
| Mac | M1 Pro/Max | 16-32GB | ~2-4s | Good for 720p output |
| Mac | M2-M4 | 32-64GB | ~1-3s | Comfortable |
| Mac | M3 Ultra | 64-192GB | ~0.8-1.5s | Best Mac performance |
| NVIDIA | RTX 4090 | 24GB | 0.328s | Original benchmark |
| NVIDIA | RTX 3080 | 10GB | ~0.5-0.8s | With xFormers |

---

## Troubleshooting

### MPS Memory Issues (OOM)

The script processes one frame at a time, so OOM should be rare. If it happens:

1. **Close other apps** - free up unified memory

2. **Set environment variables** to allow more memory usage:
   ```bash
   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python inference.py ...
   ```

---

## Pretrained Models

Pretrained models are available on [Hugging Face](https://huggingface.co/Jamichsu/Stream-DiffVSR). No manual download required - the inference script automatically fetches them.

---

## Citation

If you find this work useful, please cite the original paper:

```bibtex
@article{shiu2025stream,
  title={Stream-DiffVSR: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion},
  author={Shiu, Hau-Shiang and Lin, Chin-Yang and Wang, Zhixiang and Hsiao, Chi-Wei and Yu, Po-Fan and Chen, Yu-Chih and Liu, Yu-Lun},
  journal={arXiv preprint arXiv:2512.23709},
  year={2025}
}
```

---

## Acknowledgements
- Original implementation by [jamichss](https://github.com/jamichss/Stream-DiffVSR)
- Originally forked from [Boris Djordjevic's project](https://github.com/199-biotechnologies/stream-diffvsr)
- Built on [HuggingFace Diffusers](https://github.com/huggingface/diffusers), [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), [StableVSR](https://github.com/claudiom4sir/StableVSR), and [TAESD by Ollin](https://github.com/madebyollin/taesd)

---

## License

Apache-2.0
