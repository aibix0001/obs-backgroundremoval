#include <onnxruntime_cxx_api.h>
#include <filesystem>

#include <obs-module.h>

#include "ort-session-utils.h"
#include "consts.h"
#include "plugin-support.h"
#include "profiler.h"

static std::string getTrtCachePath()
{
	// Use a user-writable cache directory — the model data dir (/usr/share/...)
	// is root-owned and not writable by the plugin at runtime.
	const char *cacheHome = std::getenv("XDG_CACHE_HOME");
	std::filesystem::path cacheDir;
	if (cacheHome && cacheHome[0] != '\0') {
		cacheDir = std::filesystem::path(cacheHome) / "obs-backgroundremoval" / "trt-cache";
	} else {
		const char *home = std::getenv("HOME");
		if (home) {
			cacheDir = std::filesystem::path(home) / ".cache" / "obs-backgroundremoval" / "trt-cache";
		} else {
			cacheDir = std::filesystem::path("/tmp") / "obs-backgroundremoval-trt-cache";
		}
	}
	std::filesystem::create_directories(cacheDir);
	return cacheDir.string();
}

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
		if (tf->useGPU == USEGPU_TENSORRT) {
			// TensorRT V2 API with FP16, engine caching, and CUDA fallback
			try {
				std::string cachePath = getTrtCachePath();
				bool useFP16 = (tf->gpuInfo.defaultPrecision == PrecisionMode::FP16);

				obs_log(LOG_INFO, "TensorRT: cache=%s, FP16=%s", cachePath.c_str(),
					useFP16 ? "yes" : "no");

				const auto &api = Ort::GetApi();
				OrtTensorRTProviderOptionsV2 *trtOpts = nullptr;
				Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trtOpts));

				// Get model-specific TRT optimization profile shapes
				std::string profileShapes = tf->model->getTrtProfileShapes();

				std::vector<const char *> keys = {
					"device_id",
					"trt_max_workspace_size",
					"trt_fp16_enable",
					"trt_engine_cache_enable",
					"trt_engine_cache_path",
					"trt_timing_cache_enable",
					"trt_timing_cache_path",
					"trt_builder_optimization_level",
				};
				std::string fp16Str = useFP16 ? "1" : "0";
				std::vector<const char *> values = {
					"0",
					"2147483648",
					fp16Str.c_str(),
					"1",
					cachePath.c_str(),
					"1",
					cachePath.c_str(),
					"3",
				};

				// Provide explicit optimization profiles so TRT knows
				// the exact shapes for all dynamic inputs (min=opt=max).
				if (!profileShapes.empty()) {
					obs_log(LOG_INFO, "TensorRT profile shapes: %s", profileShapes.c_str());
					keys.push_back("trt_profile_min_shapes");
					values.push_back(profileShapes.c_str());
					keys.push_back("trt_profile_max_shapes");
					values.push_back(profileShapes.c_str());
					keys.push_back("trt_profile_opt_shapes");
					values.push_back(profileShapes.c_str());
				}

				Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trtOpts, keys.data(), values.data(),
										    keys.size()));
				Ort::ThrowOnError(
					api.SessionOptionsAppendExecutionProvider_TensorRT_V2(sessionOptions, trtOpts));
				api.ReleaseTensorRTProviderOptions(trtOpts);
				obs_log(LOG_INFO, "TensorRT execution provider configured");
			} catch (const std::exception &e) {
				obs_log(LOG_WARNING, "TensorRT EP failed: %s. Falling back to CUDA.", e.what());
			}
			// Always add CUDA as fallback (handles ops TensorRT doesn't support)
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
		} else {
			// CUDA execution provider
			Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
		}
		tf->session.reset(new Ort::Session(*tf->env, tf->modelFilepath.c_str(), sessionOptions));
	} catch (const std::exception &e) {
		if (tf->useGPU == USEGPU_TENSORRT) {
			// TRT can fail during session init (e.g. missing shape info on
			// intermediate nodes). Retry with CUDA-only so the filter still works.
			obs_log(LOG_WARNING, "TensorRT session failed: %s", e.what());
			obs_log(LOG_WARNING, "Retrying with CUDA-only execution provider.");
			try {
				Ort::SessionOptions cudaOptions;
				cudaOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
				cudaOptions.DisableMemPattern();
				cudaOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
				Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(cudaOptions, 0));
				tf->session.reset(new Ort::Session(*tf->env, tf->modelFilepath.c_str(), cudaOptions));
				obs_log(LOG_INFO, "CUDA fallback session created successfully");
			} catch (const std::exception &e2) {
				obs_log(LOG_ERROR, "CUDA fallback also failed: %s", e2.what());
				return OBS_BGREMOVAL_ORT_SESSION_ERROR_STARTUP;
			}
		} else {
			obs_log(LOG_ERROR, "%s", e.what());
			return OBS_BGREMOVAL_ORT_SESSION_ERROR_STARTUP;
		}
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
		tf->cudaPreprocessor.preprocess(imageBGRA.data, imageBGRA.cols, imageBGRA.rows, (int)imageBGRA.step[0],
						tf->inputTensorValues[0].data(), inputWidth, inputHeight, params);
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
