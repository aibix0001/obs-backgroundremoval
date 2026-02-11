#include "cuda-preprocess.h"

#include <cuda_runtime.h>
#include <algorithm>

// Fused BGRA→RGB resize + normalize kernel (HWC output).
// Each thread processes one output pixel.
__global__ void preprocessBGRA_HWC(const uint8_t *__restrict__ bgra, int bgraWidth, int bgraHeight, int bgraStep,
				   float *__restrict__ output, int outWidth, int outHeight, float scaleX, float scaleY,
				   float meanR, float meanG, float meanB, float invScaleR, float invScaleG,
				   float invScaleB)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= outWidth || y >= outHeight)
		return;

	// Bilinear interpolation source coordinates
	float srcX = (x + 0.5f) * scaleX - 0.5f;
	float srcY = (y + 0.5f) * scaleY - 0.5f;

	int x0 = (int)floorf(srcX);
	int y0 = (int)floorf(srcY);
	int x1 = min(x0 + 1, bgraWidth - 1);
	int y1 = min(y0 + 1, bgraHeight - 1);
	x0 = max(x0, 0);
	y0 = max(y0, 0);

	float fx = srcX - floorf(srcX);
	float fy = srcY - floorf(srcY);

	// Read 4 BGRA neighbors
	const uint8_t *p00 = bgra + y0 * bgraStep + x0 * 4;
	const uint8_t *p10 = bgra + y0 * bgraStep + x1 * 4;
	const uint8_t *p01 = bgra + y1 * bgraStep + x0 * 4;
	const uint8_t *p11 = bgra + y1 * bgraStep + x1 * 4;

	float w00 = (1.0f - fx) * (1.0f - fy);
	float w10 = fx * (1.0f - fy);
	float w01 = (1.0f - fx) * fy;
	float w11 = fx * fy;

	// BGRA layout: B=0, G=1, R=2 → output RGB
	float r = p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11;
	float g = p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11;
	float b = p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11;

	// Normalize: (pixel - mean) / scale = (pixel - mean) * invScale
	int idx = (y * outWidth + x) * 3;
	output[idx + 0] = (r - meanR) * invScaleR;
	output[idx + 1] = (g - meanG) * invScaleG;
	output[idx + 2] = (b - meanB) * invScaleB;
}

// Fused BGRA→RGB resize + normalize kernel (CHW output for BCHW models).
__global__ void preprocessBGRA_CHW(const uint8_t *__restrict__ bgra, int bgraWidth, int bgraHeight, int bgraStep,
				   float *__restrict__ output, int outWidth, int outHeight, float scaleX, float scaleY,
				   float meanR, float meanG, float meanB, float invScaleR, float invScaleG,
				   float invScaleB)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= outWidth || y >= outHeight)
		return;

	float srcX = (x + 0.5f) * scaleX - 0.5f;
	float srcY = (y + 0.5f) * scaleY - 0.5f;

	int x0 = (int)floorf(srcX);
	int y0 = (int)floorf(srcY);
	int x1 = min(x0 + 1, bgraWidth - 1);
	int y1 = min(y0 + 1, bgraHeight - 1);
	x0 = max(x0, 0);
	y0 = max(y0, 0);

	float fx = srcX - floorf(srcX);
	float fy = srcY - floorf(srcY);

	const uint8_t *p00 = bgra + y0 * bgraStep + x0 * 4;
	const uint8_t *p10 = bgra + y0 * bgraStep + x1 * 4;
	const uint8_t *p01 = bgra + y1 * bgraStep + x0 * 4;
	const uint8_t *p11 = bgra + y1 * bgraStep + x1 * 4;

	float w00 = (1.0f - fx) * (1.0f - fy);
	float w10 = fx * (1.0f - fy);
	float w01 = (1.0f - fx) * fy;
	float w11 = fx * fy;

	float r = p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11;
	float g = p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11;
	float b = p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11;

	// CHW: output[c * H * W + y * W + x]
	int planeSize = outWidth * outHeight;
	output[0 * planeSize + y * outWidth + x] = (r - meanR) * invScaleR;
	output[1 * planeSize + y * outWidth + x] = (g - meanG) * invScaleG;
	output[2 * planeSize + y * outWidth + x] = (b - meanB) * invScaleB;
}

CudaPreprocessor::~CudaPreprocessor()
{
	freeBuffers();
}

void CudaPreprocessor::ensureBuffers(size_t bgraBytes, size_t outputFloats)
{
	if (bgraBytes > bgraCapacity_) {
		if (d_bgra_)
			cudaFree(d_bgra_);
		cudaMalloc(&d_bgra_, bgraBytes);
		bgraCapacity_ = bgraBytes;
	}
	if (outputFloats > outputCapacity_) {
		if (d_output_)
			cudaFree(d_output_);
		cudaMalloc(&d_output_, outputFloats * sizeof(float));
		outputCapacity_ = outputFloats;
	}
}

void CudaPreprocessor::freeBuffers()
{
	if (d_bgra_) {
		cudaFree(d_bgra_);
		d_bgra_ = nullptr;
	}
	if (d_output_) {
		cudaFree(d_output_);
		d_output_ = nullptr;
	}
	bgraCapacity_ = 0;
	outputCapacity_ = 0;
}

void CudaPreprocessor::preprocess(const uint8_t *bgraData, int bgraWidth, int bgraHeight, int bgraStep,
				  float *outputTensor, int outWidth, int outHeight, const PreprocessParams &params)
{
	size_t bgraBytes = (size_t)bgraStep * bgraHeight;
	size_t outputFloats = (size_t)outWidth * outHeight * 3;

	ensureBuffers(bgraBytes, outputFloats);

	// Upload BGRA frame to GPU
	cudaMemcpy(d_bgra_, bgraData, bgraBytes, cudaMemcpyHostToDevice);

	// Compute resize scale factors
	float scaleX = (float)bgraWidth / (float)outWidth;
	float scaleY = (float)bgraHeight / (float)outHeight;

	// Pre-compute inverse scales for multiplication (faster than division in kernel)
	float invScaleR = (params.scaleR != 0.0f) ? 1.0f / params.scaleR : 1.0f;
	float invScaleG = (params.scaleG != 0.0f) ? 1.0f / params.scaleG : 1.0f;
	float invScaleB = (params.scaleB != 0.0f) ? 1.0f / params.scaleB : 1.0f;

	// Launch kernel
	dim3 block(16, 16);
	dim3 grid((outWidth + block.x - 1) / block.x, (outHeight + block.y - 1) / block.y);

	if (params.outputCHW) {
		preprocessBGRA_CHW<<<grid, block>>>(d_bgra_, bgraWidth, bgraHeight, bgraStep, d_output_, outWidth,
						    outHeight, scaleX, scaleY, params.meanR, params.meanG, params.meanB,
						    invScaleR, invScaleG, invScaleB);
	} else {
		preprocessBGRA_HWC<<<grid, block>>>(d_bgra_, bgraWidth, bgraHeight, bgraStep, d_output_, outWidth,
						    outHeight, scaleX, scaleY, params.meanR, params.meanG, params.meanB,
						    invScaleR, invScaleG, invScaleB);
	}

	// Download result directly to ONNX tensor buffer
	cudaMemcpy(outputTensor, d_output_, outputFloats * sizeof(float), cudaMemcpyDeviceToHost);
}
