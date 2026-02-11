# Optimization Progress

Performance optimization project targeting Ubuntu 24.04 + NVIDIA RTX GPUs.

## Phase 1: Remove Non-NVIDIA Code
- [x] Strip CPU, AMD (ROCm/MIGraphX), Intel, CoreML, DirectML execution providers
- [x] Remove macOS and Windows code paths
- [x] Keep only CUDA + TensorRT

## Phase 2a: GPU Architecture Detection
- [x] Runtime NVIDIA GPU detection via CUDA API
- [x] Architecture mapping (Turing/Ampere/Ada Lovelace)
- [x] Adaptive defaults per GPU generation

## Phase 2: Async Processing
- [x] Thread-safe inference queue (double/triple buffering)
- [x] `video_tick()` pushes frames, returns immediately
- [x] Worker thread handles inference, postprocessing on tick thread

## Phase 3: Memory Optimization
- [x] Eliminate `.clone()` copies in frame pipeline
- [x] Pre-allocate and reuse cv::Mat buffers via `copyTo()`
- [x] Eliminate full BGRA frame clone in video_tick (work inside lock scope)

## Phase 4: CUDA Preprocessing Kernels
- [x] Fused CUDA kernel: BGRA→RGB + bilinear resize + float32 + normalize
- [x] HWC and CHW output modes (for BHWC/BCHW models)
- [x] Per-model parametrized normalization (mean/scale)
- [x] Multi-arch compilation (sm_75, sm_86, sm_89)

## Phase 5: Postprocessing Optimization
- [x] Analysis: mask postprocessing operates on small data (256x256 uint8 = 65KB)
- [x] CPU is optimal for mask-size ops (L2 cache resident, no GPU transfer overhead)
- [x] Contour finding (findContours) is inherently sequential — kept on CPU
- [x] Temporal smoothing, morphological ops, blur — fast on CPU for mask dimensions

## Phase 6: TensorRT + Adaptive FP16
- [x] TensorRT V2 API with engine caching (trt-cache/ directory)
- [x] Adaptive FP16: auto-enabled for Ampere/Ada, FP32 for Turing
- [x] Timing cache for faster engine rebuilds
- [x] Graceful CUDA fallback when TensorRT SDK not installed

## Phase 7: Model Research
- [x] Evaluated RMBG v2.0 (BiRefNet-based, 1GB, 100ms+ — too slow for real-time)
- [x] Evaluated BiRefNet/BiRefNet-lite (500MB-1GB, 70-140ms — too slow)
- [x] Evaluated MODNet webcam (15MB, 3-5ms — viable but existing models sufficient)
- [x] Best real-time models already included: RVM (3-4ms), PP-HumanSeg (2-3ms), MediaPipe (<1ms)

## Phase 8: Build System Cleanup
- [x] Removed macOS/Windows CMake presets and CI workflows
- [x] Multi-arch CUDA compilation: sm_75, sm_86, sm_89
- [x] CUDAToolkit required, TensorRT graceful fallback

## Phase 9: Alpha Matte Pipeline
- [x] Added `outputsAlphaMatte()` virtual to Model base — RVM preserves continuous alpha
- [x] Skip binarization for alpha-matte models — soft edges instead of hard 0/255 cutoff
- [x] RVM default model, temporal smoothing tuned for ConvGRU

## Phase 10: Synchronous Inference for Alpha-Matte Models
- [x] Bypass async queue for RVM — run inference directly in `video_tick()`
- [x] Eliminates 2-3 frame async pipeline latency (66-100ms → 0)
- [x] ~10ms total within 33ms frame budget at 30fps

## Phase 11: Full-Resolution DGF Refiner
- [x] Feed 1080p directly to RVM with `downsample_ratio=0.25`
- [x] Backbone processes internally at 270x480 (similar compute cost)
- [x] Deep Guided Filter refiner upsamples alpha to 1080p using full-res edge guidance
- [x] Eliminates dumb bilinear upscaling — hair, fingers, clothing edges preserved
- [x] Recurrent state dimensions computed from internal resolution (ceil-div stride-2)

## Future: Standalone TensorRT + v4l2loopback Pipeline
- [ ] Native TensorRT FP16 inference (~3-5ms vs ~15-25ms through ONNX Runtime)
- [ ] V4L2 camera capture → CUDA pipeline → v4l2loopback virtual camera
- [ ] Zero-copy recurrent state ping-pong
- [ ] Works with any application (OBS, Discord, Zoom, Teams)
- [ ] See `PLAN-PATH-B-TENSORRT-V4L2LOOPBACK.md` for detailed design
