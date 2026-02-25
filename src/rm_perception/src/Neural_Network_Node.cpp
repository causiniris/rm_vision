#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include "rm_perception/yolo_detector.hpp"
#include "rm_perception/msg/armor.hpp"
#include "rm_perception/msg/armors.hpp"

using std::placeholders::_1;

namespace rm_perception {

class NeuralNetworkNode : public rclcpp::Node {
public:
    NeuralNetworkNode() : Node("neural_network_node") {
        // 请确保模型路径正确
        std::string model_path = "/home/causin/rm_vision/src/rm_perception/models/best.onnx"; 
        
        try {
            detector_ = std::make_unique<YoloDetector>(model_path);
            RCLCPP_INFO(this->get_logger(), "🧠 YOLO 模型加载成功!");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "YOLO 初始化失败: %s", e.what());
            rclcpp::shutdown();
            throw std::runtime_error("停止节点初始化");
        }

        // 【防丢帧核心】：将所有队列暴增到 1000
        rclcpp::QoS qos(1000);

        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", qos, 
            std::bind(&NeuralNetworkNode::imageCallback, this, _1));

        armors_pub_ = this->create_publisher<rm_perception::msg::Armors>("/neural_network/armors", qos);
        img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/neural_network/image_result", qos);
        
        RCLCPP_INFO(this->get_logger(), "神经网络节点已启动，正在疯狂吸入图像...");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) { return; }

        cv::Mat frame = cv_ptr->image;
        std::vector<ArmorObject> detections = detector_->detect(frame, 0.5, 0.4);

        rm_perception::msg::Armors armors_msg;
        armors_msg.header = msg->header;

        for (const auto& obj : detections) {
            rm_perception::msg::Armor armor_msg;
            armor_msg.class_id = std::to_string(obj.class_id);
            armor_msg.confidence = obj.confidence;
            armor_msg.color = (obj.class_id == 0) ? 0 : 1; 

            armor_msg.corners[0].x = obj.box.x; armor_msg.corners[0].y = obj.box.y;
            armor_msg.corners[1].x = obj.box.x; armor_msg.corners[1].y = obj.box.y + obj.box.height;
            armor_msg.corners[2].x = obj.box.x + obj.box.width; armor_msg.corners[2].y = obj.box.y + obj.box.height;
            armor_msg.corners[3].x = obj.box.x + obj.box.width; armor_msg.corners[3].y = obj.box.y;

            armor_msg.center.x = obj.box.x + obj.box.width / 2.0;
            armor_msg.center.y = obj.box.y + obj.box.height / 2.0;

            armors_msg.armors.push_back(armor_msg);

            cv::Scalar color = (armor_msg.color == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
            cv::rectangle(frame, obj.box, color, 2);
            cv::putText(frame, "ID:" + armor_msg.class_id, cv::Point(obj.box.x, obj.box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }

        armors_pub_->publish(armors_msg);
        sensor_msgs::msg::Image::SharedPtr out_img_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        img_pub_->publish(*out_img_msg);
    }

    std::unique_ptr<YoloDetector> detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Publisher<rm_perception::msg::Armors>::SharedPtr armors_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
};

} // namespace rm_perception

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<rm_perception::NeuralNetworkNode>());
    rclcpp::shutdown();
    return 0;
}