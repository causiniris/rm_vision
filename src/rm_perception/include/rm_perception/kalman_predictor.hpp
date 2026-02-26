#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

namespace rm_perception {

class KalmanPredictor {
public:
    KalmanPredictor();
    cv::Rect predict();
    cv::Rect update(const cv::Rect& measurement, bool has_measurement);
    bool isInitialized() const { return is_initialized_; }

    // 【新增】：允许外部动态修改过程噪声(Q)和测量噪声(R)
    void setNoiseParams(float q_cov, float r_cov);

private:
    cv::KalmanFilter kf_;
    bool is_initialized_;
};

} // namespace rm_perception