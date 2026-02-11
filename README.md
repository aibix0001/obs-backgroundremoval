# OBS Background Removal — NVIDIA RTX Optimized Fork

Performance-optimized fork of [obs-backgroundremoval](https://github.com/royshil/obs-backgroundremoval) targeting **Ubuntu 24.04 + NVIDIA RTX GPUs** (2000/3000/4000 series).

The upstream plugin is cross-platform but suffers from severe performance issues: synchronous processing blocks the OBS video pipeline, excessive memory copies occur every frame, and all preprocessing/postprocessing runs on the CPU. This fork strips it down to NVIDIA-only and rebuilds the pipeline for real-time performance.

## Target Hardware

| GPU Generation | Buffering | Precision | Recommended Models |
|----------------|-----------|-----------|-------------------|
| RTX 2000 (Turing, sm_75) | Double | FP32 | MediaPipe, SelfieSeg |
| RTX 3000 (Ampere, sm_86) | Double | FP16 | RVM, RMBG |
| RTX 4000 (Ada Lovelace, sm_89) | Triple | FP16 | All + TensorRT |

- **Development GPU**: RTX 2060 Super
- **OS**: Ubuntu 24.04
- **Resolution targets**: 1080p and 1440p @ 60fps

## Performance Targets

| Resolution | RTX 2000 (FP32) | RTX 3000 (FP16) | RTX 4000 (FP16+TRT) |
|------------|-----------------|-----------------|---------------------|
| 1080p @ 60fps | < 10ms/frame | < 7ms/frame | < 5ms/frame |
| 1440p @ 60fps | < 15ms/frame | < 10ms/frame | < 7ms/frame |

CPU utilization target: < 15%

## Optimization Progress

### Phase 1: Remove Non-NVIDIA Code
- [x] Strip CPU, AMD (ROCm/MIGraphX), Intel, CoreML, DirectML execution providers
- [x] Remove macOS and Windows code paths
- [x] Keep only CUDA + TensorRT

### Phase 2a: GPU Architecture Detection
- [x] Runtime NVIDIA GPU detection via CUDA API
- [x] Architecture mapping (Turing/Ampere/Ada Lovelace)
- [x] Adaptive defaults per GPU generation

### Phase 2: Async Processing
- [x] Thread-safe inference queue (double/triple buffering)
- [x] `video_tick()` pushes frames, returns immediately
- [x] Worker thread handles inference, postprocessing on tick thread

### Phase 3: Memory Optimization
- [x] Eliminate `.clone()` copies in frame pipeline
- [x] Pre-allocate and reuse cv::Mat buffers via `copyTo()`
- [x] Eliminate full BGRA frame clone in video_tick (work inside lock scope)

### Phase 4: CUDA Preprocessing Kernels
- [x] Fused CUDA kernel: BGRA→RGB + bilinear resize + float32 + normalize
- [x] HWC and CHW output modes (for BHWC/BCHW models)
- [x] Per-model parametrized normalization (mean/scale)
- [x] Multi-arch compilation (sm_75, sm_86, sm_89)

### Phase 5: Postprocessing Optimization
- [x] Analysis: mask postprocessing operates on small data (256x256 uint8 = 65KB)
- [x] CPU is optimal for mask-size ops (L2 cache resident, no GPU transfer overhead)
- [x] Contour finding (findContours) is inherently sequential — kept on CPU
- [x] Temporal smoothing, morphological ops, blur — fast on CPU for mask dimensions

### Phase 6: TensorRT + Adaptive FP16
- [x] TensorRT V2 API with engine caching (trt-cache/ directory)
- [x] Adaptive FP16: auto-enabled for Ampere/Ada, FP32 for Turing
- [x] Timing cache for faster engine rebuilds
- [x] Graceful CUDA fallback when TensorRT SDK not installed

### Phase 7: Model Research
- [x] Evaluated RMBG v2.0 (BiRefNet-based, 1GB, 100ms+ — too slow for real-time)
- [x] Evaluated BiRefNet/BiRefNet-lite (500MB-1GB, 70-140ms — too slow)
- [x] Evaluated MODNet webcam (15MB, 3-5ms — viable but existing models sufficient)
- [x] **Best real-time models already included**: RVM (3-4ms), PP-HumanSeg (2-3ms), MediaPipe (<1ms)

### Phase 8: Build System Cleanup
- [x] Removed macOS/Windows CMake presets and CI workflows (Phase 1)
- [x] Multi-arch CUDA compilation: sm_75, sm_86, sm_89 (Phase 4)
- [x] CUDAToolkit required, TensorRT graceful fallback (Phase 6)

### Phase 9: Alpha Matte Pipeline (Maxine-Quality Masking)
- [x] Downloaded RVM MobileNetV3 FP16 model (7MB, half the size of FP32)
- [x] Added `outputsAlphaMatte()` virtual to Model base — RVM preserves continuous alpha
- [x] Skip binarization for alpha-matte models — soft edges instead of hard 0/255 cutoff
- [x] Increased RVM input resolution from 320x192 to 512x288 (downsample_ratio=0.375)
- [x] Guided filter edge refinement — aligns mask edges to real image edges (hair, clothing)
- [x] Registered RVM FP16 in UI, set as default model
- [x] Tuned temporal smoothing to 0.7 (ConvGRU handles internal temporal stability)

## Building

```bash
# Ubuntu 24.04 with NVIDIA GPU
# Prerequisites: CUDA Toolkit 11.8+, TensorRT 8.6+

git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT=~/vcpkg

${VCPKG_ROOT}/vcpkg install --triplet x64-linux-obs
cmake -P cmake/DownloadOnnxruntime.cmake
cmake --preset ubuntu-ci-x86_64
cmake --build --preset ubuntu-ci-x86_64
sudo cmake --install build_x86_64
```

## Model Recommendations (by speed)

| Model | Input | Est. RTX 3060 | Quality | Best For |
|-------|-------|--------------|---------|----------|
| MediaPipe | 256x256 | <1ms | Low | Maximum FPS |
| SINet | 320x320 | <1ms | Low | Maximum FPS |
| PP-HumanSeg | 192x192 | 2-3ms | Medium | Balanced |
| RVM (MobileNetV3) | 192x320 | 3-4ms | Medium-High | Video (temporal) |
| RMBG 1.4 | 1024x1024 | 30-50ms | High | Quality priority |

All models benefit from CUDA preprocessing and TensorRT FP16 on Ampere/Ada GPUs.

Low-light enhancement models:
- TBEFN, URetinex-Net, Semantic-Guided LLIE, Zero-DCE

## Credits

This is a fork of [obs-backgroundremoval](https://github.com/royshil/obs-backgroundremoval) by [Roy Shilkrot](https://github.com/royshil) and contributors.

Original project sponsors:
- https://github.com/sponsors/royshil
- https://github.com/sponsors/umireon

Pretrained model weights for portrait segmentation:
- [SINet](https://github.com/anilsathyan7/Portrait-Segmentation/tree/master/SINet)
- [PP-HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/contrib/PP-HumanSeg)
- [MediaPipe Meet Segmentation](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/082_MediaPipe_Meet_Segmentation)
- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)
- [TCMonoDepth](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/384_TCMonoDepth)
- [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)

Image enhancement models:
- [TBEFN](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/213_TBEFN)
- [URetinex-Net](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/372_URetinex-Net)
- [Semantic-Guided LLIE](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/370_Semantic-Guided-Low-Light-Image-Enhancement)
- [Zero-DCE](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/243_Zero-DCE-improved)

Architecture walkthrough (upstream): https://youtu.be/iFQtcJg0Wsk
