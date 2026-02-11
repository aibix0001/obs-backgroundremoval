#ifndef CUDA_PREPROCESS_H
#define CUDA_PREPROCESS_H

#include <cstddef>
#include <cstdint>

// Per-channel normalization parameters for preprocessing.
// The kernel computes: output[c] = (pixel_float - mean[c]) / scale[c]
struct PreprocessParams {
	float meanR = 0.0f, meanG = 0.0f, meanB = 0.0f;
	float scaleR = 255.0f, scaleG = 255.0f, scaleB = 255.0f;
	bool outputCHW = false; // true for BCHW models
};

// CUDA-accelerated image preprocessor for ONNX model input.
// Fuses BGRA→RGB conversion, bilinear resize, float conversion, and
// normalization into a single GPU kernel launch.
class CudaPreprocessor {
public:
	CudaPreprocessor() = default;
	~CudaPreprocessor();

	CudaPreprocessor(const CudaPreprocessor &) = delete;
	CudaPreprocessor &operator=(const CudaPreprocessor &) = delete;

	// Preprocess a BGRA uint8 frame into a float32 RGB normalized tensor.
	// The output is written directly to outputTensor (CPU memory).
	// GPU buffers are allocated/resized as needed.
	void preprocess(const uint8_t *bgraData, int bgraWidth, int bgraHeight, int bgraStep, float *outputTensor,
			int outWidth, int outHeight, const PreprocessParams &params);

private:
	void ensureBuffers(size_t bgraBytes, size_t outputFloats);
	void freeBuffers();

	uint8_t *d_bgra_ = nullptr;
	float *d_output_ = nullptr;
	size_t bgraCapacity_ = 0;
	size_t outputCapacity_ = 0;
};

#endif /* CUDA_PREPROCESS_H */
