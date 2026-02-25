#include "rm_perception/traditional_detector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    std::string video_path = "/home/causin/rm_vision/videos/demo.mp4";
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << ">>> 无法打开视频，请检查路径！" << std::endl;
        return -1;
    }

    rm_perception::TraditionalDetector detector;
    cv::Mat frame;
    
    std::cout << ">>> 开始纯净版传统视觉测试，按 'ESC' 或 'q' 退出..." << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0); 
            continue;
        }

        auto start = std::chrono::high_resolution_clock::now();
        
        // 同时检测红蓝双方
        std::vector<rm_perception::TraditionalArmor> blue_armors = detector.detect(frame, 0);
        std::vector<rm_perception::TraditionalArmor> red_armors = detector.detect(frame, 1);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        // 绘制蓝色
        for (const auto& armor : blue_armors) {
            for (int i = 0; i < 4; ++i) cv::line(frame, armor.corners[i], armor.corners[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
            cv::circle(frame, armor.center, 4, cv::Scalar(0, 255, 0), -1);
        }
        // 绘制红色
        for (const auto& armor : red_armors) {
            for (int i = 0; i < 4; ++i) cv::line(frame, armor.corners[i], armor.corners[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
            cv::circle(frame, armor.center, 4, cv::Scalar(0, 255, 0), -1);
        }

        cv::putText(frame, "Inference (Dual): " + std::to_string(duration.count()).substr(0, 4) + " ms", 
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        cv::putText(frame, "Pipeline: Hardcoded Final", 
                    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Traditional Vision Final", frame);

        char key = (char)cv::waitKey(30);
        if (key == 27 || key == 'q') break;
    }

    cv::destroyAllWindows();
    return 0;
}