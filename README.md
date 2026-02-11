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
- [ ] Runtime NVIDIA GPU detection via CUDA API
- [ ] Architecture mapping (Turing/Ampere/Ada Lovelace)
- [ ] Adaptive defaults per GPU generation

### Phase 2: Async Processing
- [ ] Thread-safe inference queue (double/triple buffering)
- [ ] `video_tick()` pushes frames, returns immediately
- [ ] Worker thread handles inference and postprocessing

### Phase 3: Memory Optimization
- [ ] Eliminate `.clone()` copies in frame pipeline
- [ ] Pre-allocate and reuse cv::Mat buffers
- [ ] Move semantics / shared_ptr where appropriate

### Phase 4: CUDA Preprocessing Kernels
- [ ] BGRA to RGB conversion
- [ ] Image resize (NPP or custom kernel)
- [ ] Float conversion + normalization
- [ ] HWC to CHW transpose

### Phase 5: CUDA Postprocessing Kernels
- [ ] Thresholding (binary mask)
- [ ] Connected components (contours)
- [ ] Blur operations (Gaussian/stack blur)
- [ ] Morphological ops (erode/dilate)

### Phase 6: TensorRT + Adaptive FP16
- [ ] ONNX to TensorRT engine conversion with caching
- [ ] FP32 for Turing, FP16 for Ampere/Ada
- [ ] User-selectable precision override

### Phase 7: Model Research
- [ ] Evaluate RMBG v2, MODNet lightweight, BiRefNet
- [ ] Test INT8 quantized models
- [ ] Find TensorRT-optimized models on HuggingFace

### Phase 8: Build System Cleanup
- [ ] Remove macOS/Windows CMake presets and CI workflows
- [ ] Add multi-arch CUDA compilation (sm_75, sm_86, sm_89)
- [ ] Add CUDA Toolkit, TensorRT, NPP dependency checks

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

## Models

Background removal models:
- SINet, PP-HumanSeg, MediaPipe, RobustVideoMatting, TCMonoDepth, RMBG-1.4, Selfie Segmentation

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
