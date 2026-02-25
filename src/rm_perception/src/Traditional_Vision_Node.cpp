#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include "rm_perception/traditional_detector.hpp" // 引入你的传统视觉库
#include "rm_perception/msg/armor.hpp"
#include "rm_perception/msg/armors.hpp"

using std::placeholders::_1;

namespace rm_perception {

class TraditionalVisionNode : public rclcpp::Node {
public:
    TraditionalVisionNode() : Node("traditional_vision_node") {
        // 【防丢帧核心】：将接收和发送的队列全部设为 1000，绝不使用 SensorDataQoS
        rclcpp::QoS qos(1000);

        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", qos, 
            std::bind(&TraditionalVisionNode::imageCallback, this, _1));

        armors_pub_ = this->create_publisher<rm_perception::msg::Armors>("/traditional_vision/armors", qos);
        img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/traditional_vision/image_result", qos);

        RCLCPP_INFO(this->get_logger(), "🛡️ 传统视觉节点已启动，队列已扩容...");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            // 将 ROS 的 Image 消息转换为 OpenCV 的 cv::Mat
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge 异常: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;
        auto start = std::chrono::high_resolution_clock::now();

        // 调用我们固化好的核心算法，同时检测红蓝双方
        std::vector<TraditionalArmor> blue_armors = detector_.detect(frame, 0);
        std::vector<TraditionalArmor> red_armors = detector_.detect(frame, 1);

        auto end = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end - start).count();

        // 准备发布的自定义装甲板消息
        rm_perception::msg::Armors armors_msg;
        armors_msg.header = msg->header; // 同步时间戳极其重要，预测节点需要它来算速度！

        // 封装数据与绘制画面的 Lambda 函数
        auto process_armors = [&](const std::vector<TraditionalArmor>& armors, const cv::Scalar& color_scalar) {
            for (const auto& a : armors) {
                rm_perception::msg::Armor armor_msg;
                armor_msg.color = a.color;
                armor_msg.confidence = 1.0; // 传统视觉依靠几何硬规则，匹配上即视为100%置信
                armor_msg.class_id = "unknown"; // 传统视觉无法识别数字，暂时填未知
                
                armor_msg.center.x = a.center.x;
                armor_msg.center.y = a.center.y;

                for (int i = 0; i < 4; ++i) {
                    armor_msg.corners[i].x = a.corners[i].x;
                    armor_msg.corners[i].y = a.corners[i].y;
                    // 在原图上画框
                    cv::line(frame, a.corners[i], a.corners[(i + 1) % 4], color_scalar, 2);
                }
                cv::circle(frame, a.center, 4, cv::Scalar(0, 255, 0), -1);
                
                armors_msg.armors.push_back(armor_msg);
            }
        };

        process_armors(blue_armors, cv::Scalar(255, 0, 0));
        process_armors(red_armors, cv::Scalar(0, 0, 255));

        cv::putText(frame, "Traditional Latency: " + std::to_string(latency).substr(0, 4) + " ms",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        // 发布装甲板数据
        armors_pub_->publish(armors_msg);

        // 将画好框的 cv::Mat 转换回 ROS 消息并发布
        sensor_msgs::msg::Image::SharedPtr out_img_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        img_pub_->publish(*out_img_msg);
    }

    TraditionalDetector detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Publisher<rm_perception::msg::Armors>::SharedPtr armors_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
};
} // namespace rm_perception

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<rm_perception::TraditionalVisionNode>());
    rclcpp::shutdown();
    return 0;
}