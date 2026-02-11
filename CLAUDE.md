# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Most Important Guideline: Stay in the Project Folder

**We ALWAYS stay in the project folder.** While we can read files in other locations (e.g., headers or libs), we do not litter the filesystem outside this project. All temporary or build artifacts remain here. Missing dependencies can be installed via apt, but only after user approval.

## Project Overview

OBS Studio plugin for AI-powered background removal and low-light enhancement. Uses ONNX Runtime to run neural networks locally on Windows, macOS, and Linux.

- **Languages**: C17 (plugin-main.c), C++17 (filters, models)
- **Build System**: CMake 3.16+ with vcpkg for dependencies
- **Key Dependencies**: ONNX Runtime 1.23.2, OpenCV4, CURL, OBS Studio 31.1.1+

## Build Commands

### Ubuntu (x86_64)
```bash
# One-time setup: Install vcpkg
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT=~/vcpkg

# Install dependencies (10-20 min on first run)
${VCPKG_ROOT}/vcpkg install --triplet x64-linux-obs

# Download ONNX Runtime
cmake -P cmake/DownloadOnnxruntime.cmake

# Build
cmake --preset ubuntu-ci-x86_64
cmake --build --preset ubuntu-ci-x86_64

# Install locally for testing
sudo cmake --install build_x86_64
```

### macOS (Universal)
```bash
# Requires Xcode 16.4
./.github/scripts/install-vcpkg-macos.bash
cmake --preset macos-ci
cmake --build --preset macos-ci
cp -r build_macos/RelWithDebInfo/obs-backgroundremoval.plugin ~/Library/Application\ Support/obs-studio/plugins
```

### Windows (x64)
```powershell
# Requires Visual Studio 2022 with C++ workload
.\.github\scripts\build-ubuntu.ps1 -Target x64 -Configuration RelWithDebInfo
```

## Code Formatting (REQUIRED before commits)

**CRITICAL**: CI will fail if formatting is incorrect. Always format before committing.

```bash
# Format C/C++ files (requires clang-format 19.1.1)
./build-aux/run-clang-format

# Format CMake files (requires gersemi 0.12.0+)
./build-aux/run-gersemi

# Check without modifying
./build-aux/run-clang-format --check
./build-aux/run-gersemi --check
```

Install formatters on Ubuntu:
```bash
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
brew install obsproject/tools/clang-format@19
brew install obsproject/tools/gersemi
```

## Architecture

### Model-Based Design
The plugin uses an object-oriented model architecture. Each AI model (Selfie Segmentation, MODNet, RVM, etc.) is a separate class inheriting from `Model` or `ModelBCHW` base classes in `src/models/Model.h`.

- **Model**: Base class for BHWC (Batch-Height-Width-Channel) format models
- **ModelBCHW**: For models using BCHW format (transposed dimensions)

Key virtual methods to override when adding new models:
- `prepareInputToNetwork()` - Preprocess input (normalization, format conversion)
- `postprocessOutput()` - Postprocess output
- `getNetworkInputSize()` - Extract input dimensions from tensor shape
- `getNetworkOutput()` - Extract output Mat from tensor values

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `background-filter.cpp` | `src/` | Main background removal filter, processes video frames |
| `enhance-filter.cpp` | `src/` | Low-light enhancement filter |
| `plugin-main.c` | `src/` | OBS plugin entry point, registers filters |
| `FilterData.h` | `src/` | Main data structure for filter state |
| `ort-session-utils.cpp` | `src/ort-utils/` | ONNX Runtime session management and inference |
| `obs-utils.cpp` | `src/obs-utils/` | OBS integration utilities (config, logging) |
| `update-checker/` | `src/` | GitHub API version checking |

### Execution Flow
1. OBS calls `video_tick()` → `video_render()` for each frame
2. Frame is copied to a stage surface (`stagesurface`)
3. ONNX model runs inference via `ort-session-utils.cpp`
4. Result is rendered using shader in `data/effects/`
5. Background replacement/blending is applied

### GPU Acceleration Support

Execution providers are detected at compile time via `CheckLibraryExists` in CMakeLists.txt:
- **CUDA**: `HAVE_ONNXRUNTIME_CUDA_EP` (Linux)
- **ROCm**: `HAVE_ONNXRUNTIME_ROCM_EP` (deprecated in 1.23.0)
- **MIGraphX**: `HAVE_ONNXRUNTIME_MIGRAPHX_EP` (Linux AMD GPU alternative)
- **TensorRT**: `HAVE_ONNXRUNTIME_TENSORRT_EP` (Windows)
- **DirectML**: Windows default (always available)
- **CoreML**: macOS default (always available)

### Model Files

ONNX models are stored in `data/models/` and referenced by their filename (without extension). The `getModelFilepath()` method in `Model.h` resolves paths using `obs_module_file()`.

## Version Management

**Version is stored in `buildspec.json` - edit ONLY with `jq`:**
```bash
jq '.version = "1.2.3"' buildspec.json > buildspec.json.tmp && mv buildspec.json.tmp buildspec.json
```

Release process (when user requests "Start release"):
1. Create branch `releases/bump-X.Y.Z`
2. Update version in `buildspec.json` using `jq`
3. Create PR, merge it
4. Push git tag `X.Y.Z` (no `v` prefix) - this triggers automated release

## Testing

No automated unit tests. Manual testing required:
1. Build and install plugin locally
2. Open OBS Studio
3. Add video source (camera or file)
4. Apply "Background Removal" or "Enhance" filter
5. Verify real-time processing works

## Common Patterns

- Thread-safe frame processing uses `std::mutex inputBGRALock` and `outputLock` in `filter_data`
- Models use `obs_log()` for logging (defined in `plugin-support.h`)
- Platform-specific paths: Windows uses `std::wstring`, others use `std::string`
- All filter settings are persisted via `obs_data_t*` in OBS's config system

## Important Constraints

- macOS: No cross-architecture translation (Rosetta2) - crashes will occur
- CPU fallback works but is slower; 2-thread setting recommended
- Multiple filter instances multiply CPU/GPU usage
- Filter activation/deactivation can be controlled via settings (see "keep enabled when inactive" option)

## Workflow Rules

- **Commit often**: Make small, focused commits after each meaningful change. Don't batch unrelated changes.
- **Keep PROGRESS.md up to date**: PROGRESS.md serves as the optimization progress tracker. After completing work on any phase or sub-task, update the checkboxes. Mark items with `[x]` when done.
- **Format before committing**: Always run `./build-aux/run-clang-format` and `./build-aux/run-gersemi` before any commit.

## Active Optimization Project

**See `HANDOVER.md`** for details on the ongoing optimization project targeting Ubuntu 24.04 + NVIDIA RTX series (2000/3000/4000). Key work includes:
- Removing all non-NVIDIA code paths (CPU, AMD, Intel, macOS, Windows)
- Implementing async processing with configurable buffering
- CUDA preprocessing/postprocessing kernels
- TensorRT optimization with adaptive FP16
- GPU architecture detection for adaptive defaults
