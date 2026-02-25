#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <queue>

using std::placeholders::_1;

namespace rm_perception {

class VisualizerNode : public rclcpp::Node {
public:
    VisualizerNode() : Node("visualizer_node") {
        // 【防丢帧核心】：接收队列设为 1000
        rclcpp::QoS qos(1000);

        sub_raw_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", qos, std::bind(&VisualizerNode::rawCallback, this, _1));
        sub_trad_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/traditional_vision/image_result", qos, std::bind(&VisualizerNode::tradCallback, this, _1));
        sub_nn_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/neural_network/image_result", qos, std::bind(&VisualizerNode::nnCallback, this, _1));

        // 依然保留一个高频定时器，仅仅是为了让 OpenCV 的窗口不卡死
        ui_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30), std::bind(&VisualizerNode::uiHeartbeat, this));

        RCLCPP_INFO(this->get_logger(), "🖥️ 可视化节点启动！【纯离线不掉帧模式】已就绪...");
    }

    ~VisualizerNode() {
        if (video_writer_.isOpened()) {
            video_writer_.release();
            RCLCPP_INFO(this->get_logger(), "✅ 完美 23 秒顺滑录像已成功保存至: %s", output_path_.c_str());
        }
    }

private:
    void uiHeartbeat() { cv::waitKey(1); }

    void rawCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        raw_q_.push(cv_bridge::toCvCopy(msg, "bgr8")->image);
        tryProcess();
    }
    void tradCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        trad_q_.push(cv_bridge::toCvCopy(msg, "bgr8")->image);
        tryProcess();
    }
    void nnCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        nn_q_.push(cv_bridge::toCvCopy(msg, "bgr8")->image);
        tryProcess();
    }

    void tryProcess() {
        // 只有当三个节点都提交了作业，才进行统一批改！
        if (raw_q_.empty() || trad_q_.empty() || nn_q_.empty()) {
            return; 
        }

        cv::Mat r_raw = raw_q_.front(); raw_q_.pop();
        cv::Mat r_trad = trad_q_.front(); trad_q_.pop();
        cv::Mat r_nn = nn_q_.front(); nn_q_.pop();

        renderAndSave(r_raw, r_trad, r_nn);
    }

    void renderAndSave(cv::Mat& frame_raw, cv::Mat& frame_trad, cv::Mat& frame_nn) {
        double scale = 0.33;
        cv::Mat r_raw, r_trad, r_nn;
        cv::resize(frame_raw, r_raw, cv::Size(), scale, scale);
        cv::resize(frame_trad, r_trad, cv::Size(), scale, scale);
        cv::resize(frame_nn, r_nn, cv::Size(), scale, scale);

        cv::putText(r_raw, "1. Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(r_trad, "2. Traditional", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(r_nn, "3. YOLO Net", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::Mat stitched;
        std::vector<cv::Mat> matrices = {r_raw, r_trad, r_nn};
        cv::hconcat(matrices, stitched);

        if (!is_writer_initialized_) {
            output_path_ = "/home/causin/rm_vision/perception_demo_perfect.avi";
            int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            // 【核心】：以原视频标准的 30 FPS 保存，保证最终录像的流畅度
            video_writer_.open(output_path_, fourcc, 30.0, stitched.size());
            if (video_writer_.isOpened()) is_writer_initialized_ = true;
        }

        if (video_writer_.isOpened()) video_writer_.write(stitched);

        // 实时弹窗依然会一顿一顿的，但请无视它，后台正在生成完美的视频
        cv::imshow("Offline Rendering (Wait for it to finish...)", stitched);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_raw_, sub_trad_, sub_nn_;
    rclcpp::TimerBase::SharedPtr ui_timer_;

    std::queue<cv::Mat> raw_q_, trad_q_, nn_q_;
    cv::VideoWriter video_writer_;
    std::string output_path_;
    bool is_writer_initialized_ = false;
};

} // namespace rm_perception

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<rm_perception::VisualizerNode>());
    rclcpp::shutdown();
    return 0;
}