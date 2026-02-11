#ifndef ASYNC_INFERENCE_QUEUE_H
#define ASYNC_INFERENCE_QUEUE_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <opencv2/core.hpp>
#include <thread>

#include "gpu-info.h"

// Thread-safe async inference queue with configurable buffering.
// video_tick() pushes frames, a worker thread processes them,
// video_render() pulls the latest completed mask.
class AsyncInferenceQueue {
public:
	using InferenceFunc = std::function<bool(const cv::Mat &inputBGRA, cv::Mat &outputMask)>;

	AsyncInferenceQueue() = default;
	~AsyncInferenceQueue();

	// Start the worker thread with the given inference function and buffering mode.
	void start(InferenceFunc func, BufferingMode mode = BufferingMode::DOUBLE);

	// Stop the worker thread and clean up.
	void stop();

	// Push a new frame for processing. Non-blocking; drops frame if queue is full.
	void pushFrame(const cv::Mat &frameBGRA);

	// Get the latest completed output mask. Returns false if no mask is available.
	bool getLatestMask(cv::Mat &mask);

	// Check if the worker is running.
	bool isRunning() const { return running_.load(); }

	// Get frame processing stats.
	uint64_t framesProcessed() const { return framesProcessed_.load(); }
	uint64_t framesDropped() const { return framesDropped_.load(); }

private:
	void workerLoop();

	InferenceFunc inferenceFunc_;
	BufferingMode bufferingMode_ = BufferingMode::DOUBLE;

	std::thread workerThread_;
	std::atomic<bool> running_{false};

	// Input buffer: latest frame submitted by video_tick
	cv::Mat inputBuffer_;
	std::mutex inputMutex_;
	std::condition_variable inputCv_;
	bool hasNewInput_ = false;

	// Output buffer: latest completed mask
	cv::Mat outputBuffer_;
	std::mutex outputMutex_;
	bool hasOutput_ = false;

	// Stats
	std::atomic<uint64_t> framesProcessed_{0};
	std::atomic<uint64_t> framesDropped_{0};
};

#endif /* ASYNC_INFERENCE_QUEUE_H */
