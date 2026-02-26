#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>
#include <algorithm> 
#include <memory>
#include "rm_perception/kalman_predictor.hpp"
#include "rm_perception/msg/armors.hpp"

using std::placeholders::_1;

namespace rm_perception {

struct Track {
    KalmanPredictor kf;
    int time_since_update = 0; 
    int hit_streak = 0;        
    cv::Rect current_pred;
    int id;
};

class MultiTracker {
public:
    MultiTracker() : next_id_(1) {}

    void processAndDraw(cv::Mat& frame, const std::vector<cv::Rect>& detections) {
        for (auto& t : tracks_) {
            t->current_pred = t->kf.predict();
            t->time_since_update++;
        }

        struct MatchPair { int det_idx; int trk_idx; float dist; };
        std::vector<MatchPair> pairs;
        
        // 【修改1】：将匹配距离放宽到 150，因为旋转时目标在 2D 画面上移动得很快
        float max_dist = 150.0f; 

        for (size_t i = 0; i < detections.size(); ++i) {
            cv::Point2f det_center(detections[i].x + detections[i].width / 2.0f,
                                   detections[i].y + detections[i].height / 2.0f);
            for (size_t j = 0; j < tracks_.size(); ++j) {
                cv::Point2f trk_center(tracks_[j]->current_pred.x + tracks_[j]->current_pred.width / 2.0f,
                                       tracks_[j]->current_pred.y + tracks_[j]->current_pred.height / 2.0f);
                float dist = cv::norm(det_center - trk_center);
                if (dist < max_dist) {
                    pairs.push_back({(int)i, (int)j, dist});
                }
            }
        }

        std::sort(pairs.begin(), pairs.end(), [](const MatchPair& a, const MatchPair& b) {
            return a.dist < b.dist;
        });

        std::vector<bool> det_matched(detections.size(), false);
        std::vector<bool> trk_matched(tracks_.size(), false);

        for (const auto& p : pairs) {
            if (!det_matched[p.det_idx] && !trk_matched[p.trk_idx]) {
                tracks_[p.trk_idx]->current_pred = tracks_[p.trk_idx]->kf.update(detections[p.det_idx], true);
                tracks_[p.trk_idx]->time_since_update = 0;
                tracks_[p.trk_idx]->hit_streak++;
                det_matched[p.det_idx] = true;
                trk_matched[p.trk_idx] = true;
            }
        }

        for (size_t i = 0; i < detections.size(); ++i) {
            if (!det_matched[i]) {
                auto new_track = std::make_shared<Track>(); 
                new_track->kf.update(detections[i], true); 
                new_track->current_pred = detections[i];
                new_track->hit_streak = 1;
                new_track->id = next_id_++;
                tracks_.push_back(new_track);
            }
        }

        // 【修改2】：抗 ID 爆炸核心！将记忆寿命从 10 帧延长到 30 帧 (约 1 秒)
        // 装甲板转到背面看不见了？别急着删，等它转回来大概率还能接上旧 ID！
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
            [](const std::shared_ptr<Track>& t) { return t->time_since_update > 30; }), tracks_.end());

        // 【修改3】：抗误检核心！渲染过滤机制
        for (auto& t : tracks_) {
            // 必须连续识别到 3 帧 (hit_streak >= 3)，才认为是真正的装甲板，过滤传统视觉的闪烁误检
            // 丢失不超过 5 帧时，才在画面上画盲猜的绿框
            if (t->time_since_update < 5 && t->hit_streak >= 3) {
                cv::rectangle(frame, t->current_pred, cv::Scalar(0, 255, 0), 3);
                cv::putText(frame, "ID:" + std::to_string(t->id), 
                            cv::Point(t->current_pred.x, t->current_pred.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
            if (t->time_since_update > 0) t->hit_streak = 0; 
        }
    }

private:
    std::vector<std::shared_ptr<Track>> tracks_;
    int next_id_;
};

// ... PredictorNode 类保持不变 ...
class PredictorNode : public rclcpp::Node {
public:
    PredictorNode() : Node("predictor_node") {
        rclcpp::QoS qos(1000); 
        sub_trad_img_ = this->create_subscription<sensor_msgs::msg::Image>("/traditional_vision/image_result", qos, std::bind(&PredictorNode::tradImgCb, this, _1));
        sub_trad_armors_ = this->create_subscription<rm_perception::msg::Armors>("/traditional_vision/armors", qos, std::bind(&PredictorNode::tradArmorsCb, this, _1));
        pub_trad_img_ = this->create_publisher<sensor_msgs::msg::Image>("/predictor/traditional_image", qos);
        sub_nn_img_ = this->create_subscription<sensor_msgs::msg::Image>("/neural_network/image_result", qos, std::bind(&PredictorNode::nnImgCb, this, _1));
        sub_nn_armors_ = this->create_subscription<rm_perception::msg::Armors>("/neural_network/armors", qos, std::bind(&PredictorNode::nnArmorsCb, this, _1));
        pub_nn_img_ = this->create_publisher<sensor_msgs::msg::Image>("/predictor/neural_image", qos);
        RCLCPP_INFO(this->get_logger(), "🎯 抗旋转 & 抗误检的多目标追踪节点已启动！");
    }
private:
    void tradImgCb(const sensor_msgs::msg::Image::SharedPtr msg) { trad_img_q_.push(msg); processTrad(); }
    void tradArmorsCb(const rm_perception::msg::Armors::SharedPtr msg) { trad_armors_q_.push(msg); processTrad(); }
    void nnImgCb(const sensor_msgs::msg::Image::SharedPtr msg) { nn_img_q_.push(msg); processNN(); }
    void nnArmorsCb(const rm_perception::msg::Armors::SharedPtr msg) { nn_armors_q_.push(msg); processNN(); }
    void processTrad() {
        if (trad_img_q_.empty() || trad_armors_q_.empty()) return;
        auto img_msg = trad_img_q_.front(); trad_img_q_.pop();
        auto armors_msg = trad_armors_q_.front(); trad_armors_q_.pop();
        cv::Mat frame = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
        std::vector<cv::Rect> detections;
        for (const auto& a : armors_msg->armors) {
            detections.emplace_back(a.corners[0].x, a.corners[0].y, a.corners[2].x - a.corners[0].x, a.corners[2].y - a.corners[0].y);
        }
        tracker_trad_.processAndDraw(frame, detections);
        pub_trad_img_->publish(*cv_bridge::CvImage(img_msg->header, "bgr8", frame).toImageMsg());
    }
    void processNN() {
        if (nn_img_q_.empty() || nn_armors_q_.empty()) return;
        auto img_msg = nn_img_q_.front(); nn_img_q_.pop();
        auto armors_msg = nn_armors_q_.front(); nn_armors_q_.pop();
        cv::Mat frame = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
        std::vector<cv::Rect> detections;
        for (const auto& a : armors_msg->armors) {
            detections.emplace_back(a.corners[0].x, a.corners[0].y, a.corners[2].x - a.corners[0].x, a.corners[2].y - a.corners[0].y);
        }
        tracker_nn_.processAndDraw(frame, detections);
        pub_nn_img_->publish(*cv_bridge::CvImage(img_msg->header, "bgr8", frame).toImageMsg());
    }
    MultiTracker tracker_trad_;
    MultiTracker tracker_nn_;
    std::queue<sensor_msgs::msg::Image::SharedPtr> trad_img_q_, nn_img_q_;
    std::queue<rm_perception::msg::Armors::SharedPtr> trad_armors_q_, nn_armors_q_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_trad_img_, sub_nn_img_;
    rclcpp::Subscription<rm_perception::msg::Armors>::SharedPtr sub_trad_armors_, sub_nn_armors_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_trad_img_, pub_nn_img_;
};

} // namespace rm_perception

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<rm_perception::PredictorNode>());
    rclcpp::shutdown();
    return 0;
}