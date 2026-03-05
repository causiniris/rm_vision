#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

class CameraCalibrator {
public:
    // 构造函数新增：是否使用鱼眼模型
    CameraCalibrator(cv::Size board_size, float square_size, bool use_fisheye) 
        : board_size_(board_size), square_size_(square_size), use_fisheye_(use_fisheye) {}

    bool addImage(const cv::Mat& image) {
        std::vector<cv::Point2f> image_corners;
        bool found = cv::findChessboardCorners(image, board_size_, image_corners,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
        if (found) {
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, image_corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
            
            image_points_.push_back(image_corners);
            object_points_.push_back(createObjectPoints());
            std::cout << "✅ 抓拍成功！已收集有效视角数量: " << image_points_.size() << std::endl;
        } else {
            std::cout << "⚠️ 抓拍失败：未能清晰识别所有角点！" << std::endl;
        }
        return found;
    }

    void calibrateAndSave(const cv::Size& image_size, const std::string& save_path) {
        if (image_points_.size() < 10) {
            std::cerr << "❌ 错误：有效图像太少，请至少收集 15 张以上！" << std::endl;
            return;
        }

        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat dist_coeffs;
        std::vector<cv::Mat> rvecs, tvecs;
        double rms = 0.0;

        std::cout << "⚙️ 正在执行非线性优化计算内参，请稍候..." << std::endl;

        try {
            // 【核心架构分支】：根据用户选择调用不同的底层数学模型
            if (use_fisheye_) {
                std::cout << "🐟 启用 [Fisheye 鱼眼模型] 进行超广角解算..." << std::endl;
                // 【核心修正】：移除敏感的 CHECK_COND，并加上 FIX_PRINCIPAL_POINT (强制锁死主点在画面正中心)，极大地稳住矩阵！
                int flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_FIX_SKEW | cv::fisheye::CALIB_FIX_PRINCIPAL_POINT;
                
                rms = cv::fisheye::calibrate(object_points_, image_points_, image_size, 
                                             camera_matrix, dist_coeffs, rvecs, tvecs, flags,
                                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
            }else {
                std::cout << "📌 启用 [Pinhole 针孔模型] 进行标准解算..." << std::endl;
                rms = cv::calibrateCamera(object_points_, image_points_, image_size, 
                                          camera_matrix, dist_coeffs, rvecs, tvecs);
            }
        } catch (const cv::Exception& e) {
            std::cerr << "❌ 标定算法崩溃: " << e.what() << std::endl;
            std::cerr << "提示：鱼眼模型对标定板的姿态要求更严苛，请确保棋盘格在画面各个角落都有抓拍！" << std::endl;
            return;
        }

        std::cout << "\n========================================\n";
        std::cout << "🎯 标定完成! 投影重投影误差 (RMS): " << rms << " 像素\n";
        std::cout << "========================================\n";

        cv::FileStorage fs(save_path, cv::FileStorage::WRITE);
        fs << "model_type" << (use_fisheye_ ? "fisheye" : "pinhole"); // 写入模型类型戳
        fs << "camera_matrix" << camera_matrix;
        fs << "dist_coeffs" << dist_coeffs;
        fs.release();
        std::cout << "💾 参数已成功保存至: " << save_path << std::endl;
    }

private:
    cv::Size board_size_;
    float square_size_;
    bool use_fisheye_;
    std::vector<std::vector<cv::Point3f>> object_points_;
    std::vector<std::vector<cv::Point2f>> image_points_;

    std::vector<cv::Point3f> createObjectPoints() {
        std::vector<cv::Point3f> corners;
        for (int i = 0; i < board_size_.height; i++) {
            for (int j = 0; j < board_size_.width; j++) {
                corners.push_back(cv::Point3f(j * square_size_, i * square_size_, 0.0f));
            }
        }
        return corners;
    }
};

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  RoboTac 视觉组 - 镜头光学标定系统" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "请选择你当前使用的镜头物理模型：" << std::endl;
    std::cout << "[1] 标准针孔模型 (Pinhole) - 适用于 FOV < 120° 的普通无畸变工业相机" << std::endl;
    std::cout << "[2] 鱼眼广角模型 (Fisheye) - 适用于 FOV > 120° 的超广角/鱼眼相机" << std::endl;
    std::cout << "请输入数字 (1 或 2): ";
    
    int choice;
    std::cin >> choice;
    bool use_fisheye = (choice == 2);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开摄像头！" << std::endl;
        return -1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // 棋盘格规格
    cv::Size board_size(9, 6); 
    float square_size_mm = 15.0f; 
    CameraCalibrator calibrator(board_size, square_size_mm, use_fisheye);

    cv::Mat frame;
    int captured_count = 0;
    int empty_frame_count = 0;
    
    std::cout << "\n🚀 实时标定启动！按下 'S' 抓拍，'C' 计算并保存，'Q' 退出。" << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            empty_frame_count++;
            cv::waitKey(100); 
            if (empty_frame_count > 50) break;
            continue; 
        }
        empty_frame_count = 0; 

        cv::Mat display_frame = frame.clone();
        std::vector<cv::Point2f> corners;
        
        bool found = cv::findChessboardCorners(frame, board_size, corners, cv::CALIB_CB_FAST_CHECK);
        if (found) {
            cv::drawChessboardCorners(display_frame, board_size, corners, found);
            cv::putText(display_frame, "Ready! Press 'S' to snap.", cv::Point(20, 70), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }

        cv::putText(display_frame, "Captured: " + std::to_string(captured_count) + " / 20", 
                    cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        
        cv::imshow("Live Camera Calibration", display_frame);

        char key = (char)cv::waitKey(30);
        if (key == 'q' || key == 'Q') break;
        else if ((key == 's' || key == 'S') && found) {
            if (calibrator.addImage(frame)) {
                captured_count++;
                cv::bitwise_not(display_frame, display_frame);
                cv::imshow("Live Camera Calibration", display_frame);
                cv::waitKey(200); 
            }
        } 
        else if ((key == 'c' || key == 'C') && captured_count >= 10) {
            std::string save_path = "src/rm_perception/config/camera_params.yaml";
            calibrator.calibrateAndSave(frame.size(), save_path);
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}