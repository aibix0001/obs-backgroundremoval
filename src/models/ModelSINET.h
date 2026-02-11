#ifndef MODELSINET_H
#define MODELSINET_H

#include "Model.h"

class ModelSINET : public ModelBCHW {
public:
	ModelSINET(/* args */) {}
	~ModelSINET() {}

	virtual PreprocessParams getPreprocessParams() const
	{
		// (pixel - mean) / std, CHW output
		// cv::Scalar order is R,G,B (after BGRA→RGB conversion)
		return PreprocessParams{102.890434f, 111.25247f, 126.91212f,
					62.93292f * 255.0f, 62.82138f * 255.0f, 66.355705f * 255.0f, true};
	}

	virtual void prepareInputToNetwork(cv::Mat &resizedImage, cv::Mat &preprocessedImage)
	{
		resizedImage = (resizedImage - cv::Scalar(102.890434, 111.25247, 126.91212)) /
			       cv::Scalar(62.93292 * 255.0, 62.82138 * 255.0, 66.355705 * 255.0);
		hwc_to_chw(resizedImage, preprocessedImage);
	}

	virtual cv::Mat getNetworkOutput(const std::vector<std::vector<int64_t>> &outputDims,
					 std::vector<std::vector<float>> &outputTensorValues)
	{
		UNUSED_PARAMETER(outputDims);
		return cv::Mat(320, 320, CV_32FC2, outputTensorValues[0].data());
	}

	virtual void postprocessOutput(cv::Mat &outputImage)
	{
		cv::Mat outputTransposed;
		chw_to_hwc_32f(outputImage, outputTransposed);
		// take 2nd channel
		std::vector<cv::Mat> outputImageSplit;
		cv::split(outputTransposed, outputImageSplit);
		outputImage = outputImageSplit[1];
	}
};

#endif // MODELSINET_H
