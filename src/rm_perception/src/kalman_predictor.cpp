#include "rm_perception/kalman_predictor.hpp"

namespace rm_perception {

KalmanPredictor::KalmanPredictor() {
    // 6个状态 [cx, cy, w, h, vx, vy]，4个观测 [cx, cy, w, h]
    kf_.init(6, 4, 0);

    // 状态转移矩阵 (匀速直线运动模型)
    kf_.transitionMatrix = (cv::Mat_<float>(6, 6) <<
        1, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 1,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);

    // 测量矩阵
    kf_.measurementMatrix = (cv::Mat_<float>(4, 6) <<
        1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0);

    // 噪声协方差设置
    // 【修改后】：增大过程噪声，减小测量噪声。
    // 意思是：因为敌人在做非线性的旋转运动，我的直线预测很不可靠（噪声大），
    // 所以请滤波器尽可能相信 YOLO 传来的真实测量值！
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-2));     
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-3));
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));

    is_initialized_ = false;
}

cv::Rect KalmanPredictor::predict() {
    if (!is_initialized_) return cv::Rect();
    cv::Mat pred = kf_.predict();
    return cv::Rect(pred.at<float>(0) - pred.at<float>(2) / 2.0f,
                    pred.at<float>(1) - pred.at<float>(3) / 2.0f,
                    pred.at<float>(2), pred.at<float>(3));
}

cv::Rect KalmanPredictor::update(const cv::Rect& measurement, bool has_measurement) {
    // 首次检测到目标，强制初始化
    if (!is_initialized_ && has_measurement) {
        kf_.statePost.at<float>(0) = measurement.x + measurement.width / 2.0f;
        kf_.statePost.at<float>(1) = measurement.y + measurement.height / 2.0f;
        kf_.statePost.at<float>(2) = measurement.width;
        kf_.statePost.at<float>(3) = measurement.height;
        kf_.statePost.at<float>(4) = 0;
        kf_.statePost.at<float>(5) = 0;
        is_initialized_ = true;
        return measurement;
    }

    if (!is_initialized_) return cv::Rect();

    // 如果本帧检测到了目标，进行修正
    if (has_measurement) {
        cv::Mat meas = (cv::Mat_<float>(4, 1) <<
            measurement.x + measurement.width / 2.0f,
            measurement.y + measurement.height / 2.0f,
            measurement.width,
            measurement.height);
        kf_.correct(meas);
    }
    
    // 输出修正后的平滑状态
    cv::Mat state = kf_.statePost;
    return cv::Rect(state.at<float>(0) - state.at<float>(2) / 2.0f,
                    state.at<float>(1) - state.at<float>(3) / 2.0f,
                    state.at<float>(2), state.at<float>(3));
}

} // namespace rm_perception