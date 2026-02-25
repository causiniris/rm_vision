#ifndef RM_PERCEPTION_YOLO_DETECTOR_HPP
#define RM_PERCEPTION_YOLO_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>  // 引入 ORT C++ 接口
#include <string>
#include <vector>
#include <memory>

namespace rm_perception {

struct ArmorObject {
    int class_id;       
    float confidence;   
    cv::Rect2d box;     
};

class YoloDetector {
public:
    explicit YoloDetector(const std::string& model_path);
    ~YoloDetector() = default;

    // 【接口不变】保证外界调用的 ROS 节点和测试脚本不需要修改代码
    std::vector<ArmorObject> detect(const cv::Mat& src, float conf_threshold = 0.5, float nms_threshold = 0.4);

private:
    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;
    
    // 【核心】ONNXRuntime 引擎
    Ort::Env env_{nullptr};                       // 全局环境
    Ort::SessionOptions session_options_{nullptr};// 会话配置 (线程数、硬件加速器等)
    std::unique_ptr<Ort::Session> session_{nullptr}; // 推理核心会话
    Ort::AllocatorWithDefaultOptions allocator_;  // 内存分配器

    // 输入输出节点的名称和维度
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    std::vector<int64_t> input_node_dims_;

    cv::Mat formatToSquare(const cv::Mat& source);
};

} // namespace rm_perception

#endif // RM_PERCEPTION_YOLO_DETECTOR_HPP