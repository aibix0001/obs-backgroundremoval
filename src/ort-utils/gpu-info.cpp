#include "gpu-info.h"

#include <cuda_runtime.h>
#include <obs-module.h>
#include "plugin-support.h"

bool detectGpu(GpuInfo &info)
{
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err != cudaSuccess || deviceCount == 0) {
		obs_log(LOG_WARNING, "No CUDA-capable GPU detected: %s", cudaGetErrorString(err));
		return false;
	}

	cudaDeviceProp props;
	err = cudaGetDeviceProperties(&props, 0);
	if (err != cudaSuccess) {
		obs_log(LOG_ERROR, "Failed to get GPU properties: %s", cudaGetErrorString(err));
		return false;
	}

	info.deviceId = 0;
	info.name = props.name;
	info.computeCapabilityMajor = props.major;
	info.computeCapabilityMinor = props.minor;
	info.totalMemoryMB = props.totalGlobalMem / (1024 * 1024);

	int sm = props.major * 10 + props.minor;

	if (sm >= 89) {
		info.architecture = GpuArchitecture::ADA_LOVELACE;
		info.defaultBuffering = BufferingMode::TRIPLE;
		info.defaultPrecision = PrecisionMode::FP16;
	} else if (sm >= 80) {
		info.architecture = GpuArchitecture::AMPERE;
		info.defaultBuffering = BufferingMode::DOUBLE;
		info.defaultPrecision = PrecisionMode::FP16;
	} else if (sm >= 75) {
		info.architecture = GpuArchitecture::TURING;
		info.defaultBuffering = BufferingMode::DOUBLE;
		info.defaultPrecision = PrecisionMode::FP32;
	} else {
		info.architecture = GpuArchitecture::UNKNOWN;
		info.defaultBuffering = BufferingMode::DOUBLE;
		info.defaultPrecision = PrecisionMode::FP32;
	}

	obs_log(LOG_INFO, "GPU detected: %s (sm_%d, %zuMB, %s)", info.name.c_str(), sm, info.totalMemoryMB,
		gpuArchitectureName(info.architecture));

	return true;
}

const char *gpuArchitectureName(GpuArchitecture arch)
{
	switch (arch) {
	case GpuArchitecture::TURING:
		return "Turing";
	case GpuArchitecture::AMPERE:
		return "Ampere";
	case GpuArchitecture::ADA_LOVELACE:
		return "Ada Lovelace";
	default:
		return "Unknown";
	}
}
