#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

namespace rm_perception {

class KalmanPredictor {
public:
    KalmanPredictor();
    
    // 预测步骤：卡尔曼自带的惯性预测
    cv::Rect predict();
    
    // 更新步骤：用检测到的真实框去修正预测
    cv::Rect update(const cv::Rect& measurement, bool has_measurement);
    
    // 供外部获取初始化状态
    bool isInitialized() const { return is_initialized_; }

private:
    cv::KalmanFilter kf_;
    bool is_initialized_;
};

} // namespace rm_perception