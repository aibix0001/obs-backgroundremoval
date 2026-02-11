#ifndef MODELTCMONODEPTH_H
#define MODELTCMONODEPTH_H

#include "Model.h"

class ModelTCMonoDepth : public ModelBCHW {
private:
	/* data */
public:
	ModelTCMonoDepth(/* args */) {}
	~ModelTCMonoDepth() {}

	virtual PreprocessParams getPreprocessParams() const
	{
		// No normalization, just CHW transpose (values stay in [0, 255] range)
		return PreprocessParams{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, true};
	}

	virtual void prepareInputToNetwork(cv::Mat &resizedImage, cv::Mat &preprocessedImage)
	{
		// Do not normalize from [0, 255] to [0, 1].

		hwc_to_chw(resizedImage, preprocessedImage);
	}

	virtual void postprocessOutput(cv::Mat &outputImage)
	{
		cv::normalize(outputImage, outputImage, 1.0, 0.0, cv::NORM_MINMAX);
	}
};

#endif // MODELTCMONODEPTH_H
