#ifndef MODELRVM_H
#define MODELRVM_H

#include "Model.h"

class ModelRVM : public ModelBCHW {
private:
	// Model input resolution — the ONNX model supports dynamic shapes,
	// so the CUDA preprocessor resizes the source frame to this size.
	// With downsample_ratio < 1, the model internally processes at a lower
	// resolution and the Deep Guided Filter refiner upsamples the alpha
	// matte back to this size using the full-res source for edge guidance.
	static constexpr int INPUT_WIDTH = 1920;
	static constexpr int INPUT_HEIGHT = 1080;
	static constexpr float DOWNSAMPLE_RATIO = 0.25f;

	// Channel counts for the 4 ConvGRU recurrent states
	static constexpr int REC_CHANNELS[4] = {16, 20, 40, 64};

public:
	ModelRVM(/* args */) {}
	~ModelRVM() {}

	virtual bool outputsAlphaMatte() const { return true; }

	virtual void populateInputOutputNames(const std::unique_ptr<Ort::Session> &session,
					      std::vector<Ort::AllocatedStringPtr> &inputNames,
					      std::vector<Ort::AllocatedStringPtr> &outputNames)
	{
		Ort::AllocatorWithDefaultOptions allocator;

		inputNames.clear();
		outputNames.clear();

		for (size_t i = 0; i < session->GetInputCount(); i++) {
			inputNames.push_back(session->GetInputNameAllocated(i, allocator));
		}
		// Skip output[0] (fgr) — we only need pha + recurrent states
		for (size_t i = 1; i < session->GetOutputCount(); i++) {
			outputNames.push_back(session->GetOutputNameAllocated(i, allocator));
		}
	}

	virtual bool populateInputOutputShapes(const std::unique_ptr<Ort::Session> &session,
					       std::vector<std::vector<int64_t>> &inputDims,
					       std::vector<std::vector<int64_t>> &outputDims)
	{
		inputDims.clear();
		outputDims.clear();

		for (size_t i = 0; i < session->GetInputCount(); i++) {
			const Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(i);
			const auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
			inputDims.push_back(inputTensorInfo.GetShape());
		}

		for (size_t i = 1; i < session->GetOutputCount(); i++) {
			const Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(i);
			const auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
			outputDims.push_back(outputTensorInfo.GetShape());
		}

		// src input: full resolution (the DGF refiner uses this for edge guidance)
		inputDims[0][0] = 1;
		inputDims[0][2] = INPUT_HEIGHT;
		inputDims[0][3] = INPUT_WIDTH;

		// Recurrent state dimensions are at backbone stride fractions of the
		// INTERNAL resolution (after downsample_ratio is applied by the model).
		// MobileNetV3 backbone uses stride-2 convolutions: ceil(dim/2) per stage.
		int internal_h = (int)(INPUT_HEIGHT * DOWNSAMPLE_RATIO);
		int internal_w = (int)(INPUT_WIDTH * DOWNSAMPLE_RATIO);
		int h = internal_h;
		int w = internal_w;

		for (int i = 0; i < 4; i++) {
			h = (h + 1) / 2; // ceil division by 2 (stride-2 conv)
			w = (w + 1) / 2;
			// Input recurrent states (r1i..r4i at indices 1..4)
			inputDims[i + 1][0] = 1;
			inputDims[i + 1][1] = REC_CHANNELS[i];
			inputDims[i + 1][2] = h;
			inputDims[i + 1][3] = w;
		}
		// downsample_ratio input (index 5): shape [1], already correct from model

		// pha output: full resolution (DGF refiner upsamples to match src)
		outputDims[0][0] = 1;
		outputDims[0][2] = INPUT_HEIGHT;
		outputDims[0][3] = INPUT_WIDTH;

		// Recurrent state outputs (same dims as inputs)
		h = internal_h;
		w = internal_w;
		for (int i = 0; i < 4; i++) {
			h = (h + 1) / 2;
			w = (w + 1) / 2;
			outputDims[i + 1][0] = 1;
			outputDims[i + 1][2] = h;
			outputDims[i + 1][3] = w;
		}

		return true;
	}

	virtual void setExtraTensorInputs(std::vector<std::vector<float>> &inputTensorValues)
	{
		inputTensorValues[5][0] = DOWNSAMPLE_RATIO;
	}

	virtual void loadInputToTensor(const cv::Mat &preprocessedImage, uint32_t, uint32_t,
				       std::vector<std::vector<float>> &inputTensorValues)
	{
		inputTensorValues[0].assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());
		inputTensorValues[5][0] = DOWNSAMPLE_RATIO;
	}

	virtual void assignOutputToInput(std::vector<std::vector<float>> &outputTensorValues,
					 std::vector<std::vector<float>> &inputTensorValues)
	{
		for (size_t i = 1; i < 5; i++) {
			inputTensorValues[i].assign(outputTensorValues[i].begin(), outputTensorValues[i].end());
		}
	}
};

#endif /* MODELRVM_H */
