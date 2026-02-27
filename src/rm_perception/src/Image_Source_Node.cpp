#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

class ImageSourceNode : public rclcpp::Node {
public:
    ImageSourceNode() : Node("image_source_node") {
        // 【防丢帧核心】：将发送队列拉满到 1000
        rclcpp::QoS qos(1000);
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", qos);
        
        // 确保路径是你真实的 23 秒视频路径
        video_path_ = "videos/demo.mp4"; 
        cap_.open(video_path_);
        
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "无法打开视频文件: %s", video_path_.c_str());
            rclcpp::shutdown();
            return;
        }

        RCLCPP_INFO(this->get_logger(), "🚀 图像源节点启动！(进入离线不掉帧渲染模式)");

        // 每 250ms 发一帧，给 YOLO 留出 0.25 秒的运算时间
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(250), 
            std::bind(&ImageSourceNode::timerCallback, this));
    }

private:
    void timerCallback() {
        cv::Mat frame;
        cap_ >> frame;

        if (frame.empty()) {
            RCLCPP_INFO(this->get_logger(), "🎉 23秒原视频全部发送完毕！请等待可视化窗口处理完最后几帧，然后按 Ctrl+C 结束。");
            timer_->cancel(); // 发完就停，不再循环
            return;
        }

        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "camera_optical_frame";
        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
        pub_->publish(*msg);
    }

    cv::VideoCapture cap_;
    std::string video_path_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageSourceNode>());
    rclcpp::shutdown();
    return 0;
}