# Optimization Plan: OBS Background Removal for Ubuntu + NVIDIA RTX Series

## Context

The current OBS background removal plugin has severe performance issues:
- Synchronous processing blocks the entire OBS video pipeline
- Excessive memory copies (frames cloned, masks cloned every frame)
- CPU-bound preprocessing/postprocessing
- No GPU acceleration for OpenCV operations
- Multiple format conversions (BGRA→RGB→float→BHWC/BCHW)

Target: Ubuntu 24.04 + NVIDIA RTX series (2000, 3000, 4000 series)
- **Development target**: RTX 2060 Super (Turing, sm_75)
- **Optimization target**: RTX 4090 (Ada Lovelace, sm_89)
- Adaptive optimizations based on GPU architecture detected at runtime

## Recommended Approach

### Phase 1: Remove Non-NVIDIA Code (Cleanup)

**Files to modify:**

1. `src/consts.h` - Remove execution provider constants
   - Remove: `USEGPU_ROCM`, `USEGPU_MIGRAPHX`, `USEGPU_COREML`, `USEGPU_CPU`
   - Keep: `USEGPU_CUDA`, `USEGPU_TENSORRT`

2. `src/ort-utils/ort-session-utils.cpp` - Remove non-NVIDIA EP code
   - Remove ROCm, MIGraphX, CoreML preprocessor blocks
   - Remove CPU threading options (`SetInterOpNumThreads`, `SetIntraOpNumThreads`)

3. `src/background-filter.cpp` - Remove UI options for non-NVIDIA EPs
   - Remove dropdown entries for ROCm, MIGraphX, CoreML, CPU
   - Keep only CUDA and TensorRT options

4. `CMakeLists.txt` - Simplify execution provider detection
   - Keep CUDA and TensorRT checks only
   - Remove ROCm, MIGraphX checks

5. `CMakePresets.json` - Remove non-Ubuntu presets
   - Remove: `macos`, `macos-ci`, `windows-x64`, `windows-ci-x64`
   - Keep: `ubuntu-x86_64`, `ubuntu-ci-x86_64`

6. `cmake/DownloadOnnxruntime.cmake` - Simplify for Linux CUDA builds only
   - Remove macOS and Windows download logic
   - Focus on CUDA-enabled ONNX Runtime

### Phase 2: Implement Async Processing

**Critical bottleneck:** `video_tick()` runs synchronously, blocking OBS.

**New file: `src/ort-utils/async-inference-queue.{h,cpp}`**
- Create a thread-safe inference queue using `std::queue` and mutexes
- Worker thread that continuously processes frames
- Configurable buffering strategy:
  - **Triple buffer**: Input buffer, processing buffer, output buffer. Smoothest playback, adds ~1 frame latency. Default for RTX 3000/4000.
  - **Double buffer**: Front buffer (current) + back buffer (processing). Lower latency, more risk of frame drops. Default for RTX 2000.
- User-selectable via OBS filter settings (with smart defaults per GPU)

**GPU Architecture Detection (new file: `src/ort-utils/gpu-info.{h,cpp}`):**
- Detect NVIDIA GPU at runtime using CUDA
- Map GPU to architecture (Turing, Ampere, Ada Lovelace)
- Set appropriate defaults:
  - RTX 2000: Double buffer, FP32, smaller models (MediaPipe, SelfieSeg)
  - RTX 3000: Double buffer, FP16, medium models (RVM, RMBG)
  - RTX 4000: Triple buffer, FP16, all models + TensorRT

**Modify: `src/background-filter.cpp`**
- `video_tick()` pushes frame to queue, returns immediately
- `video_render()` pulls latest completed mask
- Worker thread handles all inference and postprocessing

**File: `src/FilterData.h`**
- Add inference queue and thread management fields
- Add buffering mode enum and configuration
- Add GPU architecture detection field

### Phase 3: Optimize Memory Usage

**Current issue:** Multiple `.clone()` calls per frame create expensive copies.

**Modify: `src/obs-utils/obs-utils.cpp`**
- `getRGBAFromStageSurface()`: Remove unnecessary clone, use reference

**Modify: `src/background-filter.cpp` (video_tick)**
- Remove `imageBGRA.clone()` - use move semantics or shared_ptr
- Remove `lastImageBGRA.clone()` for PSNR - use reference comparison
- Remove `backgroundMask.clone()` in temporal smoothing - use in-place operations

**New approach:**
- Pre-allocate all cv::Mat buffers during initialization
- Reuse buffers across frames
- Use `cv::Mat::operator=` for shallow copies when possible

### Phase 4: GPU-Accelerated Preprocessing (CUDA Kernels)

**Current issue:** OpenCV preprocessing runs on CPU.

**New file: `src/ort-utils/cuda-preprocessing.{h,cu,cpp}`**
- Custom CUDA kernels for:
  - BGRA → RGB conversion
  - Image resize (using NPP library or custom kernel)
  - Float conversion and normalization (divide by 255.0)
  - HWC → CHW transpose (for BCHW models)

**Modify: `src/models/Model.h`**
- Add `virtual bool preprocessWithCUDA(const cv::Mat& input, std::vector<float>& output)`
- Subclasses override to enable GPU preprocessing path

### Phase 5: CUDA Postprocessing Kernels

**Current issue:** Expensive CPU operations (contour finding, multiple blurs).

**New file: `src/ort-utils/cuda-postprocessing.{h,cu,cpp}`**
- CUDA kernels for:
  - Thresholding (binary mask creation)
  - Contour filtering (connected components on GPU)
  - Blur operations (Gaussian, stack blur equivalent)
  - Morphological operations (erode, dilate)
  - Feather/blur for mask edges

**Modify: `src/background-filter.cpp`**
- Replace CPU `findContours()` with CUDA connected components
- Replace `cv::stackBlur()`, `cv::erode()`, `cv::dilate()` with CUDA equivalents
- Keep fallback to CPU for debugging

### Phase 6: TensorRT Optimization with Adaptive FP16

**Enable TensorRT for maximum performance:**

**Modify: `cmake/DownloadOnnxruntime.cmake`**
- Download TensorRT-enabled ONNX Runtime build
- Or build ONNX Runtime with TensorRT support

**New file: `src/ort-utils/tensorrt-utils.{h,cpp}`**
- ONNX model conversion to TensorRT engine
- Cache TensorRT engines to disk for faster startup
- **Adaptive FP16:**
  - RTX 2000: FP32 only (Turing has limited FP16 throughput)
  - RTX 3000: FP16 enabled (Ampere has good FP16)
  - RTX 4000: FP16 enabled (Ada has excellent FP16)
- FP32 fallback option for all GPUs (user selectable)

**Modify: `src/ort-utils/ort-session-utils.cpp`**
- Add TensorRT-specific session options with runtime GPU detection:
  ```cpp
  #ifdef HAVE_ONNXRUNTIME_TENSORRT_EP
      if (tf->useGPU == USEGPU_TENSORRT) {
          // Detect GPU architecture and set FP16 accordingly
          bool useFP16 = (gpuArchitecture >= AMPERE); // RTX 3000+
          OrtTensorRTProviderOptionsV2* trt_options = nullptr;
          // Configure for FP16/FP32, workspace size, etc.
          OrtSessionOptionsAppendExecutionProvider_TensorRT_V2(...);
      }
  #endif
  ```

### Phase 7: Model Research and Addition

**Research HuggingFace for faster models:**
- Look for models optimized for real-time inference
- Candidate types:
  - Quantized models (INT8)
  - Smaller models with similar accuracy
  - TensorRT-optimized models
  - Models designed for GPU acceleration

**Models to research:**
- RMBG v2 (if available)
- MODNet lightweight variants
- BiRefNet (latest state-of-the-art)
- Any models with TensorRT optimization

**Add new models:**
- Create new model class in `src/models/`
- Register in `background_filter_update()`
- Add to UI properties

### Phase 8: Build System Cleanup

**Modify: `CMakeLists.txt`**
- Remove Windows/macOS specific code blocks
- Simplify vcpkg triplet to just `x64-linux-obs`
- Add CUDA and TensorRT dependency checks:
  - Find CUDA Toolkit (11.8+ for RTX 4000 support)
  - Find TensorRT (8.6+ for Ada Lovelace support)
  - Find NPP (NVIDIA Performance Primitives) for resize
- Add OpenCV CUDA modules (`opencv_cudawarping`, `opencv_cudaimgproc`)
- Set CUDA architectures for multi-generational support:
  - sm_75 (RTX 2060/2070/2080 - Turing)
  - sm_86 (RTX 3070/3080/3090 - Ampere)
  - sm_89 (RTX 4080/4090 - Ada Lovelace)
- Add .cu file compilation rule
- Add FP16 support detection

**Modify: `CMakePresets.json`**
- Keep only Ubuntu presets
- Add CUDA architecture flags (sm_86 for RTX 3090/4090, sm_75 for RTX 2060 Super)

**Modify: `.github/workflows/`**
- Remove macOS and Windows CI jobs
- Focus CI on Ubuntu only
- Add CUDA/TensorRT validation in CI

## Implementation Order

1. **Phase 1** (Cleanup) - Prerequisite for everything else
2. **Phase 2a** (GPU detection) - Needed before adaptive defaults
3. **Phase 2** (Async) - Biggest performance impact
4. **Phase 3** (Memory) - Significant improvement, low risk
5. **Phase 6** (TensorRT + adaptive FP16) - GPU optimization
6. **Phase 4** (CUDA preprocessing) - Complement to TensorRT
7. **Phase 5** (CUDA postprocessing) - Fine-tuning
8. **Phase 7** (Models) - Ongoing research
9. **Phase 8** (Build cleanup) - Can be done alongside

## Verification

**Testing on RTX 2060 Super (Turing - sm_75):**
1. Build plugin on Ubuntu 24.04
2. Install to OBS test environment
3. Test with camera source at 1080p 30fps and 60fps
4. Verify double-buffering is default
5. Verify FP32 mode is default
6. Monitor CPU/GPU utilization
7. Check for frame drops
8. Test all models (focus on lighter ones: MediaPipe, SelfieSeg)

**Testing on RTX 4090 (Ada Lovelace - sm_89):**
1. Same tests as above
2. Verify triple-buffering is default
3. Enable FP16 mode
4. Enable TensorRT optimization
6. Test all models including heavy ones (RMBG, RVM)
7. Compare performance metrics with RTX 2060

**Performance targets:**

| Resolution | RTX 2000 (FP32) | RTX 3000 (FP16) | RTX 4000 (FP16+TRT) |
|------------|-----------------|-----------------|---------------------|
| 1080p @ 60fps | < 10ms/frame | < 7ms/frame | < 5ms/frame |
| 1440p @ 60fps | < 15ms/frame | < 10ms/frame | < 7ms/frame |

**CPU utilization target:** < 15% (should be mostly idle with GPU acceleration)

**Files to modify summary:**
- `src/consts.h` - Remove non-NVIDIA constants, add GPU architecture enum
- `src/FilterData.h` - Add async queue fields, GPU architecture field
- `src/background-filter.cpp` - Async processing, adaptive defaults, remove CPU options
- `src/enhance-filter.cpp` - Similar changes
- `src/ort-utils/ort-session-utils.cpp` - Remove non-NVIDIA EPs, add TensorRT config with FP16 detection
- `src/ort-utils/ort-session-utils.h` - Update interface
- `src/obs-utils/obs-utils.cpp` - Optimize memory usage
- `src/models/Model.h` - Add GPU preprocessing hooks
- `CMakeLists.txt` - Simplify for Ubuntu only, add multi-arch CUDA support
- `CMakePresets.json` - Remove non-Ubuntu presets
- `cmake/DownloadOnnxruntime.cmake` - Linux CUDA only
- `.github/workflows/*.yaml` - Remove non-Ubuntu CI

**New files to create:**
- `src/ort-utils/async-inference-queue.h`
- `src/ort-utils/async-inference-queue.cpp`
- `src/ort-utils/gpu-info.h`
- `src/ort-utils/gpu-info.cpp`
- `src/ort-utils/cuda-preprocessing.h`
- `src/ort-utils/cuda-preprocessing.cu`
- `src/ort-utils/cuda-postprocessing.h`
- `src/ort-utils/cuda-postprocessing.cu`
- `src/ort-utils/tensorrt-utils.h`
- `src/ort-utils/tensorrt-utils.cpp`
