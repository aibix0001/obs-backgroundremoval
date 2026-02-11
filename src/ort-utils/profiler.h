#ifndef PROFILER_H
#define PROFILER_H

#ifdef ENABLE_NVTX_PROFILING
#include <nvtx3/nvToolsExt.h>

// RAII NVTX range for automatic push/pop scoping
class NvtxRange {
public:
	explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
	~NvtxRange() { nvtxRangePop(); }

	NvtxRange(const NvtxRange &) = delete;
	NvtxRange &operator=(const NvtxRange &) = delete;
};

// Color-coded NVTX range for visual distinction in Nsight
class NvtxColorRange {
public:
	NvtxColorRange(const char *name, uint32_t color)
	{
		nvtxEventAttributes_t attr = {};
		attr.version = NVTX_VERSION;
		attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		attr.colorType = NVTX_COLOR_ARGB;
		attr.color = color;
		attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
		attr.message.ascii = name;
		nvtxRangePushEx(&attr);
	}
	~NvtxColorRange() { nvtxRangePop(); }

	NvtxColorRange(const NvtxColorRange &) = delete;
	NvtxColorRange &operator=(const NvtxColorRange &) = delete;
};

#define NVTX_RANGE(name) NvtxRange _nvtx_##__LINE__(name)
#define NVTX_RANGE_COLOR(name, color) NvtxColorRange _nvtx_c_##__LINE__(name, color)

// Predefined colors for pipeline stages
#define NVTX_COLOR_TICK 0xFF00FF00     // Green: video_tick
#define NVTX_COLOR_RENDER 0xFF0000FF   // Blue: video_render
#define NVTX_COLOR_PREPROCESS 0xFFFF8000  // Orange: preprocessing
#define NVTX_COLOR_INFERENCE 0xFFFF0000   // Red: inference
#define NVTX_COLOR_POSTPROCESS 0xFFFF00FF // Magenta: postprocessing
#define NVTX_COLOR_MEMCOPY 0xFFFFFF00    // Yellow: memory copies

#else

// No-op when profiling is disabled
#define NVTX_RANGE(name)
#define NVTX_RANGE_COLOR(name, color)
#define NVTX_COLOR_TICK 0
#define NVTX_COLOR_RENDER 0
#define NVTX_COLOR_PREPROCESS 0
#define NVTX_COLOR_INFERENCE 0
#define NVTX_COLOR_POSTPROCESS 0
#define NVTX_COLOR_MEMCOPY 0

#endif // ENABLE_NVTX_PROFILING

#endif /* PROFILER_H */
