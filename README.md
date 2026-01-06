# Stream-DiffVSR: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion

**Original Authors:** Hau-Shiang Shiu, Chin-Yang Lin, Zhixiang Wang, Chi-Wei Hsiao, Po-Fan Yu, Yu-Chih Chen, Yu-Lun Liu

<a href='https://jamichss.github.io/stream-diffvsr-project-page/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://huggingface.co/Jamichsu/Stream-DiffVSR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20(v1)-blue"></a> &nbsp;
<a href="https://arxiv.org/abs/2512.23709"><img src="https://img.shields.io/badge/arXiv-2510.12747-b31b1b.svg"></a>

---

## Apple Silicon / Mac Optimization

**This fork has been optimized for Apple Silicon (M1/M2/M3/M4) Macs using Metal Performance Shaders (MPS).**

### What's Changed

| Component | Original | Mac-Optimized |
|-----------|----------|---------------|
| Device | CUDA only | Auto-detect (MPS/CUDA/CPU) |
| Memory | xFormers | Attention slicing + VAE slicing |
| Dependencies | NVIDIA CUDA stack | Minimal PyTorch MPS |
| Platform | Linux + NVIDIA GPU | macOS + Apple Silicon |

### Key Optimizations

- **Automatic device detection** - Seamlessly uses MPS on Mac, CUDA on NVIDIA, or CPU as fallback
- **Attention slicing** - Reduces memory usage by ~40% on unified memory
- **VAE slicing** - Processes VAE in chunks for large images
- **Device-agnostic code** - All hardcoded CUDA references removed
- **Streamlined dependencies** - Removed 50+ NVIDIA-specific packages

---

## Abstract

Diffusion-based video super-resolution (VSR) methods achieve strong perceptual quality but remain impractical for latency-sensitive settings due to reliance on future frames and expensive multi-step denoising. We propose Stream-DiffVSR, a causally conditioned diffusion framework for efficient online VSR. Operating strictly on past frames, it combines a four-step distilled denoiser for fast inference, an Auto-regressive Temporal Guidance (ARTG) module injecting motion-aligned cues during latent denoising, and a lightweight temporal-aware decoder with a Temporal Processor Module (TPM) enhancing detail and temporal coherence.

**Performance:**
- RTX 4090 (CUDA): 0.328s per 720p frame
- M1/M2/M3 (MPS): ~1-3s per 720p frame (depending on chip and RAM)

---

## Installation

### Mac / Apple Silicon (Recommended for this fork)

```bash
git clone https://github.com/199-biotechnologies/stream-diffvsr.git
cd stream-diffvsr

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Mac-optimized dependencies
pip install -r requirements-mac.txt
```

**Requirements:**
- macOS 12.3+ (Monterey or later)
- Python 3.9+
- Apple Silicon Mac (M1/M2/M3/M4)
- 16GB+ RAM recommended (8GB minimum)

### Linux / NVIDIA GPU (Original)

```bash
git clone https://github.com/199-biotechnologies/stream-diffvsr.git
cd stream-diffvsr
conda env create -f requirements.yml
conda activate stream-diffvsr
```

---

## Usage

### Basic Inference

```bash
python inference.py \
    --model_id 'Jamichsu/Stream-DiffVSR' \
    --out_path './output/' \
    --in_path './input/' \
    --num_inference_steps 4
```

The script automatically detects and uses the best available device (MPS on Mac, CUDA on NVIDIA).

### Force Specific Device

```bash
# Force MPS (Apple Silicon)
python inference.py --device mps --in_path ./input/ --out_path ./output/

# Force CPU (slower, but works everywhere)
python inference.py --device cpu --in_path ./input/ --out_path ./output/

# Force CUDA (NVIDIA GPUs)
python inference.py --device cuda --in_path ./input/ --out_path ./output/
```

### Input Format

The model expects input organized as subdirectories containing sequential PNG frames:

```
YOUR_INPUT_PATH/
├── video1/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
├── video2/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
```

### TensorRT Acceleration (NVIDIA Only)

For additional acceleration on NVIDIA GPUs using TensorRT:

```bash
python inference.py \
    --model_id 'Jamichsu/Stream-DiffVSR' \
    --out_path './output/' \
    --in_path './input/' \
    --num_inference_steps 4 \
    --enable_tensorrt \
    --image_height 720 \
    --image_width 1280
```

**Note:** TensorRT is not available on Mac/MPS and will be automatically disabled.

---

## Expected Performance

| Device | Chip | RAM | 720p Frame Time | Notes |
|--------|------|-----|-----------------|-------|
| Mac | M1 | 8GB | ~4-6s | May need lower resolution |
| Mac | M1 Pro/Max | 16-32GB | ~2-4s | Good for 720p |
| Mac | M2/M3 Pro/Max | 32-64GB | ~1-3s | Comfortable |
| Mac | M3 Ultra | 64-192GB | ~0.8-1.5s | Best Mac performance |
| NVIDIA | RTX 4090 | 24GB | 0.328s | Original benchmark |
| NVIDIA | RTX 3080 | 10GB | ~0.5-0.8s | With xFormers |

---

## Troubleshooting

### MPS Memory Issues

If you encounter memory errors on Mac:

1. **Reduce resolution** - Process at 540p instead of 720p
2. **Close other apps** - Free up unified memory
3. **Use CPU fallback** - `--device cpu` (slower but stable)

### "MPS backend out of memory"

```bash
# Set environment variable to enable memory fallback
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python inference.py ...
```

### Unsupported Operations

Some PyTorch operations may not be supported on MPS. The script will automatically fall back to CPU for those operations. You can enable explicit fallback:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python inference.py ...
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
- Mac/MPS optimization by [199 Biotechnologies](https://github.com/199-biotechnologies)
- Built on [HuggingFace Diffusers](https://github.com/huggingface/diffusers), [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), [StableVSR](https://github.com/claudiom4sir/StableVSR), and [TAESD](https://github.com/madebyollin/taesd)

---

## License

Apache-2.0
