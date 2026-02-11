#include <onnxruntime_cxx_api.h>
#include <filesystem>

#include <obs-module.h>

#include "ort-session-utils.h"
#include "consts.h"
#include "plugin-support.h"
#include "profiler.h"

int createOrtSession(filter_data *tf)
{
	if (tf->model.get() == nullptr) {
		obs_log(LOG_ERROR, "Model object is not initialized");
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_INVALID_MODEL;
	}

	Ort::SessionOptions sessionOptions;

	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	sessionOptions.DisableMemPattern();
	sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

	char *modelFilepath_rawPtr = obs_module_file(tf->modelSelection.c_str());

	if (modelFilepath_rawPtr == nullptr) {
		obs_log(LOG_ERROR, "Unable to get model filename %s from plugin.", tf->modelSelection.c_str());
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_FILE_NOT_FOUND;
	}

	std::string modelFilepath_s(modelFilepath_rawPtr);

	tf->modelFilepath = std::string(modelFilepath_rawPtr);

	bfree(modelFilepath_rawPtr);

	try {
		if (tf->useGPU == USEGPU_CUDA) {
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
		}
		if (tf->useGPU == USEGPU_TENSORRT) {
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0));
		}
		tf->session.reset(new Ort::Session(*tf->env, tf->modelFilepath.c_str(), sessionOptions));
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "%s", e.what());
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_STARTUP;
	}

	Ort::AllocatorWithDefaultOptions allocator;

	tf->model->populateInputOutputNames(tf->session, tf->inputNames, tf->outputNames);

	if (!tf->model->populateInputOutputShapes(tf->session, tf->inputDims, tf->outputDims)) {
		obs_log(LOG_ERROR, "Unable to get model input and output shapes");
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_INVALID_INPUT_OUTPUT;
	}

	for (size_t i = 0; i < tf->inputNames.size(); i++) {
		obs_log(LOG_INFO, "Model %s input %d: name %s shape (%d dim) %d x %d x %d x %d",
			tf->modelSelection.c_str(), (int)i, tf->inputNames[i].get(), (int)tf->inputDims[i].size(),
			(int)tf->inputDims[i][0], ((int)tf->inputDims[i].size() > 1) ? (int)tf->inputDims[i][1] : 0,
			((int)tf->inputDims[i].size() > 2) ? (int)tf->inputDims[i][2] : 0,
			((int)tf->inputDims[i].size() > 3) ? (int)tf->inputDims[i][3] : 0);
	}
	for (size_t i = 0; i < tf->outputNames.size(); i++) {
		obs_log(LOG_INFO, "Model %s output %d: name %s shape (%d dim) %d x %d x %d x %d",
			tf->modelSelection.c_str(), (int)i, tf->outputNames[i].get(), (int)tf->outputDims[i].size(),
			(int)tf->outputDims[i][0], ((int)tf->outputDims[i].size() > 1) ? (int)tf->outputDims[i][1] : 0,
			((int)tf->outputDims[i].size() > 2) ? (int)tf->outputDims[i][2] : 0,
			((int)tf->outputDims[i].size() > 3) ? (int)tf->outputDims[i][3] : 0);
	}

	// Allocate buffers
	tf->model->allocateTensorBuffers(tf->inputDims, tf->outputDims, tf->outputTensorValues, tf->inputTensorValues,
					 tf->inputTensor, tf->outputTensor);

	return OBS_BGREMOVAL_ORT_SESSION_SUCCESS;
}

bool runFilterModelInference(filter_data *tf, const cv::Mat &imageBGRA, cv::Mat &output)
{
	if (tf->session.get() == nullptr) {
		return false;
	}
	if (tf->model.get() == nullptr) {
		return false;
	}

	uint32_t inputWidth, inputHeight;
	tf->model->getNetworkInputSize(tf->inputDims, inputWidth, inputHeight);

	// CUDA-accelerated preprocessing: BGRA→RGB + resize + normalize + optional CHW
	// Writes directly to ONNX tensor buffer, replacing cvtColor/resize/convertTo/prepareInput/loadInput
	{
		NVTX_RANGE_COLOR("cuda_preprocess", NVTX_COLOR_PREPROCESS);
		PreprocessParams params = tf->model->getPreprocessParams();
		tf->cudaPreprocessor.preprocess(imageBGRA.data, imageBGRA.cols, imageBGRA.rows,
						(int)imageBGRA.step[0], tf->inputTensorValues[0].data(), inputWidth,
						inputHeight, params);
	}

	// Set model-specific extra tensor inputs (e.g., RVM downsample flag)
	tf->model->setExtraTensorInputs(tf->inputTensorValues);

	// Run network inference
	{
		NVTX_RANGE_COLOR("model_inference", NVTX_COLOR_INFERENCE);
		tf->model->runNetworkInference(tf->session, tf->inputNames, tf->outputNames, tf->inputTensor,
					       tf->outputTensor);
	}

	// Get output
	cv::Mat outputImage = tf->model->getNetworkOutput(tf->outputDims, tf->outputTensorValues);

	// Assign output to input in some models that have temporal information
	tf->model->assignOutputToInput(tf->outputTensorValues, tf->inputTensorValues);

	// Post-process output
	{
		NVTX_RANGE_COLOR("postprocess_output", NVTX_COLOR_POSTPROCESS);
		tf->model->postprocessOutput(outputImage);
	}

	// Convert [0,1] float to CV_8U [0,255]
	outputImage.convertTo(output, CV_8U, 255.0);

	return true;
}
