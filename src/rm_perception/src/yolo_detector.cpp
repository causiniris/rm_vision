#include "rm_perception/yolo_detector.hpp"
#include <stdexcept>
#include <iostream>

namespace rm_perception {

YoloDetector::YoloDetector(const std::string& model_path) {
    try {
        // 1. 初始化 ORT 环境 (开启警告级别日志)
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "RoboTac_YOLOv8");
        
        // 2. 配置会话参数
        session_options_ = Ort::SessionOptions();
        session_options_.SetIntraOpNumThreads(4); // 设置 CPU 推理线程数为 4
        // 【进阶扩展区】未来如果你把代码移到带显卡的工控机上，只需在这里加两行代码就能开启 CUDA 加速！
        // OrtCUDAProviderOptions cuda_options;
        // session_options_.AppendExecutionProvider_CUDA(cuda_options);

        // 3. 加载 ONNX 模型
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        // 4. 写死 YOLOv8 默认的输入输出节点名称 
        // (如果你的模型由标准 ultralytics 导出，必定是这两个名字)
        input_node_names_ = {"images"};
        output_node_names_ = {"output0"};

        // 设置输入张量的维度 [Batch_Size, Channels, Height, Width]
        input_node_dims_ = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};

    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ONNXRuntime 初始化彻底失败: " + std::string(e.what()));
    }
}

cv::Mat YoloDetector::formatToSquare(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

std::vector<ArmorObject> YoloDetector::detect(const cv::Mat& src, float conf_threshold, float nms_threshold) {
    if (src.empty()) {
        throw std::invalid_argument("传入图像为空！");
    }

    std::vector<ArmorObject> detections;
    cv::Mat input_image = formatToSquare(src);

    // ==========================================
    // 步骤 A: 前处理与张量构建 (指针与内存对齐)
    // ==========================================
    cv::Mat blob;
    // 依然借用 OpenCV 的 blobFromImage 来完成 BGR->RGB, 归一化和 NCHW 内存重排
    cv::dnn::blobFromImage(input_image, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    // 【硬核内存操作】创建一个 CPU 内存描述符
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // 【零拷贝思想】直接将 blob.data 的指针绑定到 ORT 的 Tensor 上，绝不进行深拷贝
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        (float*)blob.data,         // 强制转换为 float* 数据指针
        blob.total(),              // 元素总数 (1*3*640*640)
        input_node_dims_.data(),   // 维度数组
        input_node_dims_.size()    // 维度数量 (4维)
    );

    // ==========================================
    // 步骤 B: 引擎推理
    // ==========================================
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_node_names_.data(), 
            &input_tensor, 
            1, 
            output_node_names_.data(), 
            1
        );
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ORT 前向传播崩溃: " + std::string(e.what()));
    }

    // ==========================================
    // 步骤 C: 后处理与高维张量解析
    // ==========================================
    // 获取输出的裸数据指针 (raw_output 包含了 1x6x8400 个浮点数)
    float* raw_output = output_tensors[0].GetTensorMutableData<float>();

    // 【绝妙技巧】将裸指针重新包裹回 cv::Mat，利用 OpenCV 极速完成矩阵转置
    // 原始输出是 6行(属性) x 8400列(预测框)
    cv::Mat output(6, 8400, CV_32F, raw_output);
    // 转置为 8400行 x 6列，方便按行遍历
    cv::Mat predictions = output.t(); 

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    float x_factor = input_image.cols / (float)INPUT_WIDTH;
    float y_factor = input_image.rows / (float)INPUT_HEIGHT;

    for (int i = 0; i < predictions.rows; ++i) {
        float* data = predictions.ptr<float>(i);
        
        // data[4] 是蓝色置信度, data[5] 是红色置信度
        cv::Mat scores(1, 2, CV_32F, data + 4); 
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

        if (max_class_score > conf_threshold) {
            confidences.push_back(max_class_score);
            class_ids.push_back(class_id_point.x);

            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];
            
            int left = int((cx - 0.5 * w) * x_factor);
            int top = int((cy - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_indices);

    for (int idx : nms_indices) {
        ArmorObject obj;
        obj.class_id = class_ids[idx];
        obj.confidence = confidences[idx];
        obj.box = boxes[idx];
        detections.push_back(obj);
    }

    return detections;
}

} // namespace rm_perception