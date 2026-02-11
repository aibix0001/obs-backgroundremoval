#include "async-inference-queue.h"

#include <obs-module.h>

#include "profiler.h"
#include "plugin-support.h"

AsyncInferenceQueue::~AsyncInferenceQueue()
{
	stop();
}

void AsyncInferenceQueue::start(InferenceFunc func, BufferingMode mode)
{
	if (running_.load()) {
		stop();
	}

	inferenceFunc_ = std::move(func);
	bufferingMode_ = mode;
	running_.store(true);
	hasNewInput_ = false;
	hasOutput_ = false;
	framesProcessed_.store(0);
	framesDropped_.store(0);

	workerThread_ = std::thread(&AsyncInferenceQueue::workerLoop, this);

	obs_log(LOG_INFO, "Async inference started (%s buffering)",
		mode == BufferingMode::TRIPLE ? "triple" : "double");
}

void AsyncInferenceQueue::stop()
{
	if (!running_.load()) {
		return;
	}

	running_.store(false);
	inputCv_.notify_all();

	if (workerThread_.joinable()) {
		workerThread_.join();
	}

	obs_log(LOG_INFO, "Async inference stopped (processed: %llu, dropped: %llu)",
		(unsigned long long)framesProcessed_.load(), (unsigned long long)framesDropped_.load());
}

void AsyncInferenceQueue::pushFrame(const cv::Mat &frameBGRA)
{
	NVTX_RANGE_COLOR("async_push_frame", NVTX_COLOR_MEMCOPY);

	std::lock_guard<std::mutex> lock(inputMutex_);

	if (hasNewInput_) {
		// Previous frame wasn't consumed yet — drop it
		framesDropped_.fetch_add(1);
	}

	frameBGRA.copyTo(inputBuffer_);
	hasNewInput_ = true;
	inputCv_.notify_one();
}

bool AsyncInferenceQueue::getLatestMask(cv::Mat &mask)
{
	std::lock_guard<std::mutex> lock(outputMutex_);
	if (!hasOutput_ || outputBuffer_.empty()) {
		return false;
	}
	// Swap instead of copy — caller gets the buffer, we release it
	cv::swap(outputBuffer_, mask);
	hasOutput_ = false;
	return true;
}

void AsyncInferenceQueue::workerLoop()
{
	cv::Mat localInput;
	cv::Mat localOutput;

	while (running_.load()) {
		// Wait for new input
		{
			std::unique_lock<std::mutex> lock(inputMutex_);
			inputCv_.wait(lock, [this] { return hasNewInput_ || !running_.load(); });

			if (!running_.load()) {
				break;
			}

			// Swap input buffer into local — avoids 8.3MB copy,
			// lock is held so pushFrame can't race
			cv::swap(inputBuffer_, localInput);
			hasNewInput_ = false;
		}

		if (localInput.empty()) {
			continue;
		}

		// Run inference
		{
			NVTX_RANGE_COLOR("async_inference_worker", NVTX_COLOR_INFERENCE);

			if (inferenceFunc_ && inferenceFunc_(localInput, localOutput)) {
				// Publish result
				std::lock_guard<std::mutex> lock(outputMutex_);
				cv::swap(localOutput, outputBuffer_);
				hasOutput_ = true;
				framesProcessed_.fetch_add(1);
			}
		}
	}
}
