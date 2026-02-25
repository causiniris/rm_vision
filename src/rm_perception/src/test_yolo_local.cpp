// 文件路径: rm_vision/src/rm_perception/src/test_yolo_local.cpp

#include "rm_perception/yolo_detector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    // 【注意】为了确保你能一把跑通，这里我们先使用硬编码的绝对路径。
    // 请将下面的路径替换为你 Ubuntu 虚拟机中的真实绝对路径！
    // 例如："/home/你的用户名/rt_vision/rm_vision/src/rm_perception/models/best.onnx"
    std::string model_path = "/home/causin/rm_vision/src/rm_perception/models/best.onnx";
    std::string video_path = "/home/causin/rm_vision/videos/demo.mp4";

    try {
        std::cout << ">>> 正在加载 ONNX 模型..." << std::endl;
        rm_perception::YoloDetector detector(model_path);
        std::cout << ">>> 模型加载成功！" << std::endl;

        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("无法打开测试视频，请检查路径！");
        }

        cv::Mat frame;
        std::cout << ">>> 开始处理视频流，按 'ESC' 或 'q' 退出..." << std::endl;

        while (true) {
            cap >> frame;
            if (frame.empty()) {
                std::cout << ">>> 视频播放结束，循环播放..." << std::endl;
                cap.set(cv::CAP_PROP_POS_FRAMES, 0); // 循环播放视频方便调试
                continue;
            }

            // 记录推理时间
            auto start = std::chrono::high_resolution_clock::now();
            
            // 核心调用：执行神经网络推理 (置信度阈值0.5，NMS阈值0.4)
            auto armors = detector.detect(frame, 0.5, 0.4);
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> inference_time = end - start;

            // 渲染可视化结果
            for (const auto& armor : armors) {
                // 类别 0(Blue)画蓝框，1(Red)画红框 (OpenCV是BGR通道)
                cv::Scalar color = (armor.class_id == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
                cv::rectangle(frame, armor.box, color, 2);
                
                std::string label = (armor.class_id == 0 ? "Blue " : "Red ") + std::to_string(armor.confidence).substr(0, 4);
                cv::putText(frame, label, cv::Point(armor.box.x, armor.box.y - 10), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            }

            cv::putText(frame, "Inference: " + std::to_string(inference_time.count()).substr(0, 4) + " ms", 
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            cv::imshow("YOLO Algorithm Debug (Local)", frame);
            
            char key = (char)cv::waitKey(1); // 如果想看慢动作，可以把 1 改成 30
            if (key == 27 || key == 'q') break;
        }
    } catch (const std::exception& e) {
        std::cerr << ">>> 调试程序发生致命错误: " << e.what() << std::endl;
        return -1;
    }

    cv::destroyAllWindows();
    return 0;
}