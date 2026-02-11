#ifndef GPU_INFO_H
#define GPU_INFO_H

#include <cstdint>
#include <string>

enum class GpuArchitecture {
	UNKNOWN = 0,
	TURING = 75,       // RTX 2000 series (sm_75)
	AMPERE = 86,       // RTX 3000 series (sm_86)
	ADA_LOVELACE = 89, // RTX 4000 series (sm_89)
};

enum class BufferingMode {
	DOUBLE = 2, // Lower latency, default for Turing
	TRIPLE = 3, // Smoother playback, default for Ampere/Ada
};

enum class PrecisionMode {
	FP32 = 0, // Default for Turing
	FP16 = 1, // Default for Ampere/Ada
};

struct GpuInfo {
	std::string name;
	int deviceId = 0;
	int computeCapabilityMajor = 0;
	int computeCapabilityMinor = 0;
	size_t totalMemoryMB = 0;
	GpuArchitecture architecture = GpuArchitecture::UNKNOWN;
	BufferingMode defaultBuffering = BufferingMode::DOUBLE;
	PrecisionMode defaultPrecision = PrecisionMode::FP32;
};

// Detect the first NVIDIA GPU and return its info.
// Returns false if no NVIDIA GPU is found.
bool detectGpu(GpuInfo &info);

// Get a human-readable string for the GPU architecture.
const char *gpuArchitectureName(GpuArchitecture arch);

#endif /* GPU_INFO_H */
