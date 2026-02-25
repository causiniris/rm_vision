#ifndef RM_PERCEPTION_TRADITIONAL_DETECTOR_HPP
#define RM_PERCEPTION_TRADITIONAL_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace rm_perception {

struct LightBar {
    cv::RotatedRect rect;
    double length;
    double angle;
    cv::Point2f center;
};

struct TraditionalArmor {
    cv::Point2f corners[4]; // 左上, 左下, 右下, 右上
    cv::Point2f center;
    int color; // 0: Blue, 1: Red
};

class TraditionalDetector {
public:
    TraditionalDetector() = default;
    ~TraditionalDetector() = default;

    // 核心接口
    std::vector<TraditionalArmor> detect(const cv::Mat& src, int enemy_color);
    
    // 获取二值化图像供节点可视化使用
    cv::Mat getBinImage() const { return bin_image_; }

private:
    cv::Mat bin_image_;

    // 固化的通用几何参数
    const double min_light_area_ = 10.0;   // 远距离微小灯条阈值
    const double max_light_angle_ = 25.0;  // 真实灯条最大倾斜角
    const double max_angle_diff_ = 8.0;    // 两灯条最大平行度误差
    const double min_armor_ratio_ = 1.2;   // 最小装甲板长宽比
    const double max_armor_ratio_ = 4.5;   // 最大装甲板长宽比

    cv::Mat preprocess(const cv::Mat& src, int enemy_color);
    std::vector<LightBar> findLights(const cv::Mat& bin_img);
    std::vector<TraditionalArmor> matchLights(const std::vector<LightBar>& lights, int enemy_color, const cv::Mat& bin_img);
};

} // namespace rm_perception

#endif // RM_PERCEPTION_TRADITIONAL_DETECTOR_HPP