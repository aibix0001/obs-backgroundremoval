# Optimization Project Handover Document

## Project Overview

**Goal**: Optimize OBS Background Removal plugin for Ubuntu 24.04 + NVIDIA RTX series GPUs

**Current State**: Plugin is unusably slow due to:
- Synchronous processing blocking OBS video pipeline
- Excessive memory copies (frames/masks cloned every frame)
- CPU-bound preprocessing/postprocessing
- No GPU acceleration for OpenCV operations

**Targets**:
- **Development**: RTX 2060 Super (Turing, sm_75)
- **Optimization**: RTX 4090 (Ada Lovelace, sm_89)
- **Support**: Full RTX lineup (2000, 3000, 4000 series)

## User Requirements (Confirmed)

1. **Remove ALL non-NVIDIA code**: Drop Intel, AMD, macOS, Windows, CPU fallback, ROCm, MIGraphX, CoreML
2. **Execution Providers**: Keep CUDA + TensorRT only
3. **Models**: Keep all existing models AND research/add better ones from HuggingFace
4. **Preprocessing**: Custom CUDA kernels (BGRA→RGB, resize, normalize, HWC→CHW)
5. **Postprocessing**: CUDA kernels (threshold, contours, blur, morphological ops)
6. **Async Processing**: Configurable buffering (double for RTX 2000, triple for RTX 3000/4000)
7. **TensorRT**: Adaptive FP16 (FP32 for Turing, FP16 for Ampere/Ada)

## Architecture Strategy

### Adaptive Defaults per GPU Generation

| GPU Generation | Buffering | Precision | Models |
|----------------|-----------|-----------|--------|
| RTX 2000 (Turing) | Double | FP32 | Light (MediaPipe, SelfieSeg) |
| RTX 3000 (Ampere) | Double | FP16 | Medium (RVM, RMBG) |
| RTX 4000 (Ada) | Triple | FP16 | All + TensorRT |

### Key Files to Modify

**Core Changes:**
- `src/consts.h` - Remove non-NVIDIA EP constants, add GPU architecture enum
- `src/FilterData.h` - Add async queue, GPU architecture fields
- `src/background-filter.cpp` - Async processing, adaptive defaults
- `src/enhance-filter.cpp` - Same changes as background-filter
- `src/ort-utils/ort-session-utils.cpp` - TensorRT config with runtime FP16 detection
- `src/obs-utils/obs-utils.cpp` - Remove unnecessary clones
- `src/models/Model.h` - Add GPU preprocessing hooks

**Build System:**
- `CMakeLists.txt` - Multi-arch CUDA (sm_75, sm_86, sm_89), CUDA/TensorRT deps
- `CMakePresets.json` - Remove macOS/Windows presets
- `cmake/DownloadOnnxruntime.cmake` - Linux CUDA only
- `.github/workflows/` - Remove non-Ubuntu CI

### New Files to Create

1. **`src/ort-utils/async-inference-queue.{h,cpp}`**
   - Thread-safe inference queue
   - Worker thread for continuous processing
   - Double/triple buffer support

2. **`src/ort-utils/gpu-info.{h,cpp}`**
   - Runtime GPU detection using CUDA
   - Architecture mapping (Turing/Ampere/Ada)
   - Default settings per GPU generation

3. **`src/ort-utils/cuda-preprocessing.{h,cu,cpp}`**
   - BGRA → RGB kernel
   - Resize kernel (NPP or custom)
   - Float conversion + normalization
   - HWC → CHW transpose

4. **`src/ort-utils/cuda-postprocessing.{h,cu,cpp}`**
   - Thresholding
   - Connected components (contours)
   - Blur (Gaussian/stack blur equivalent)
   - Morphological ops (erode/dilate)

5. **`src/ort-utils/tensorrt-utils.{h,cpp}`**
   - ONNX → TensorRT engine conversion
   - Engine caching
   - FP16 configuration based on GPU

## Implementation Order

1. **Phase 1**: Cleanup (remove non-NVIDIA code)
2. **Phase 2a**: GPU detection infrastructure
3. **Phase 2**: Async processing (biggest impact)
4. **Phase 3**: Memory optimization (remove clones)
5. **Phase 6**: TensorRT + adaptive FP16
6. **Phase 4**: CUDA preprocessing
7. **Phase 5**: CUDA postprocessing
8. **Phase 7**: Model research
9. **Phase 8**: Build cleanup (can parallel)

## Performance Targets

| Resolution | RTX 2000 (FP32) | RTX 3000 (FP16) | RTX 4000 (FP16+TRT) |
|------------|-----------------|-----------------|---------------------|
| 1080p @ 60fps | < 10ms/frame | < 7ms/frame | < 5ms/frame |
| 1440p @ 60fps | < 15ms/frame | < 10ms/frame | < 7ms/frame |

**CPU target**: < 15% utilization

## Important Ground Rules

From `CLAUDE.md`:
> We ALWAYS stay in the project folder. While we can read files in other locations (e.g., headers or libs), we do not litter the filesystem outside this project. All temporary or build artifacts remain here. Missing dependencies can be installed via apt, but only after user approval.

## Dependencies to Install (with approval)

- CUDA Toolkit 11.8+
- TensorRT 8.6+
- NPP (NVIDIA Performance Primitives)
- OpenCV CUDA modules (opencv_cudawarping, opencv_cudaimgproc)

## Testing Approach

1. Build on Ubuntu 24.04 with RTX 2060 Super first
2. Verify 1080p @ 60fps works without drops
3. Then test on RTX 4090 with 4K
4. Monitor CPU/GPU utilization with `nvidia-smi` and `htop`

## Models to Research on HuggingFace

- RMBG v2
- MODNet lightweight variants
- BiRefNet (SOTA)
- Any TensorRT-optimized models
- INT8 quantized models

## Current Code Understanding

**Key bottleneck locations:**
- `src/background-filter.cpp:504-670` - `video_tick()` synchronous processing
- `src/ort-utils/ort-session-utils.cpp:123-170` - CPU-bound inference
- `src/obs-utils/obs-utils.cpp:30-71` - Frame cloning

**Execution provider selection:** `src/ort-utils/ort-session-utils.cpp:58-85`
**Model factory:** `src/background-filter.cpp:320-343`
**Frame processing:** `src/background-filter.cpp:546-656`

## Next Steps

Start with Phase 1 (Cleanup) - removing non-NVIDIA code paths. This is the prerequisite for all other work.

1. Edit `src/consts.h` - remove ROCm, MIGraphX, CoreML, CPU constants
2. Edit `src/ort-utils/ort-session-utils.cpp` - remove non-NVIDIA EP blocks
3. Edit `src/background-filter.cpp` - remove non-NVIDIA UI options
4. Edit `CMakeLists.txt` - simplify EP detection
5. Edit `CMakePresets.json` - remove macOS/Windows presets
6. Edit `cmake/DownloadOnnxruntime.cmake` - Linux CUDA only

Then move to Phase 2a (GPU detection) before implementing async processing.
