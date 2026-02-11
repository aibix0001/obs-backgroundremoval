# OBS Background Removal — NVIDIA RTX Optimized

Real-time AI background removal plugin for OBS Studio, optimized for NVIDIA RTX GPUs on Linux.

Uses [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) (RVM) with the Deep Guided Filter refiner to produce broadcast-quality soft alpha mattes with clean edges around hair, fingers, and clothing — no green screen needed.

## Features

- **Soft alpha matte output** — continuous transparency, not binary cutoffs
- **Temporal consistency** — RVM's ConvGRU recurrent state eliminates flickering between frames
- **Full-resolution edge refinement** — Deep Guided Filter uses the 1080p source for edge-aware upsampling
- **Synchronous inference** — mask tracks movements in real-time with zero pipeline latency
- **CUDA-accelerated preprocessing** — fused BGRA→RGB + resize + normalize kernel
- **TensorRT support** — adaptive FP16 with engine caching for Ampere/Ada GPUs
- **GPU auto-detection** — optimal defaults per RTX generation (Turing, Ampere, Ada Lovelace)

## Supported Hardware

| GPU Generation | Architecture | Precision | Status |
|----------------|-------------|-----------|--------|
| RTX 2000 (Turing) | sm_75 | FP32 | Tested (dev GPU: RTX 2060 Super) |
| RTX 3000 (Ampere) | sm_86 | FP16 | Supported |
| RTX 4000 (Ada Lovelace) | sm_89 | FP16 | Supported |

**OS**: Ubuntu 24.04 (other Linux distros may work but are untested)
**Resolution**: 1080p and 1440p @ 30-60fps

## Installation

### Prerequisites

```bash
# CUDA Toolkit 11.8+
sudo apt install nvidia-cuda-toolkit

# cuDNN 9 (via pip, then symlink)
pip install nvidia-cudnn-cu12
sudo ln -sf $(python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib/libcudnn*.so* \
    /usr/lib/x86_64-linux-gnu/

# vcpkg (C++ package manager)
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT=~/vcpkg
```

### Build from Source

```bash
git clone https://github.com/aibix0001/obs-backgroundremoval.git
cd obs-backgroundremoval

# Install C++ dependencies (first run takes 10-20 minutes)
${VCPKG_ROOT}/vcpkg install --triplet x64-linux-obs

# Download ONNX Runtime 1.23.2 GPU
cmake -P cmake/DownloadOnnxruntime.cmake

# Build
cmake --preset ubuntu-ci-x86_64
cmake --build --preset ubuntu-ci-x86_64

# Install to system OBS plugins directory
sudo cmake --install build_x86_64
```

### Verify Installation

1. Launch OBS Studio
2. Add a video source (camera or capture card)
3. Right-click the source → **Filters** → **+** → **Background Removal**
4. The default model (Robust Video Matting) should activate immediately

## Configuration

The filter works out of the box with sensible defaults. For advanced tuning, enable **Advanced** in the filter properties:

| Setting | Default | Description |
|---------|---------|-------------|
| Model | Robust Video Matting | Best quality. Other models available for speed. |
| Inference Device | CUDA | Use TensorRT for faster inference on Ampere/Ada |
| Blur Background | 0 | Set 1-20 for background blur instead of removal |
| Temporal Smooth | 0.7 | Higher = more stable, lower = more responsive |

### Available Models

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| **Robust Video Matting** | ~20ms | Excellent | Soft alpha, temporal consistency, DGF refiner |
| PP-HumanSeg | ~3ms | Medium | Fast binary segmentation |
| MediaPipe | <1ms | Low | Ultra-fast, low quality |
| SINet | <1ms | Low | Ultra-fast |
| RMBG 1.4 | ~40ms | High | Best static quality, no temporal |
| Selfie Segmentation | ~2ms | Medium | Google's selfie model |

## Performance

Measured on RTX 2060 Super at 1080p:

| Stage | Time |
|-------|------|
| CUDA preprocess (BGRA→RGB + normalize) | ~1.8ms |
| RVM inference (1080p, downsample_ratio=0.25) | ~15-20ms |
| Total per frame | ~20ms |
| GPU utilization | ~70% |

Faster GPUs (RTX 3070+) will have significant headroom for 60fps.

## Credits

Fork of [obs-backgroundremoval](https://github.com/royshil/obs-backgroundremoval) by [Roy Shilkrot](https://github.com/royshil) and contributors.

Pretrained models:
- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) (MobileNetV3) — Lin et al., WACV 2022
- [SINet](https://github.com/anilsathyan7/Portrait-Segmentation/tree/master/SINet)
- [PP-HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/contrib/PP-HumanSeg)
- [MediaPipe Meet Segmentation](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/082_MediaPipe_Meet_Segmentation)
- [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) — BRIA AI
- [TCMonoDepth](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/384_TCMonoDepth)

Low-light enhancement: TBEFN, URetinex-Net, Semantic-Guided LLIE, Zero-DCE

## License

GPL-3.0 — see [LICENSE](LICENSE)
