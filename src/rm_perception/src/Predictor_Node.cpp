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

// 【升级版】支持独立调参的多目标追踪器
class MultiTracker {
public:
    // 构造函数接收专属参数：过程噪声、测量噪声、最大允许丢失帧数、最少确认帧数、最大匹配距离
    MultiTracker(float q, float r, int max_age, int min_hits, float max_dist) 
        : q_cov_(q), r_cov_(r), max_age_(max_age), min_hits_(min_hits), max_dist_(max_dist), next_id_(1) {}

    void processAndDraw(cv::Mat& frame, const std::vector<cv::Rect>& detections, const cv::Scalar& color, const std::string& label) {
        for (auto& t : tracks_) {
            t->current_pred = t->kf.predict();
            t->time_since_update++;
        }

        struct MatchPair { int det_idx; int trk_idx; float dist; };
        std::vector<MatchPair> pairs;

        for (size_t i = 0; i < detections.size(); ++i) {
            cv::Point2f det_center(detections[i].x + detections[i].width / 2.0f,
                                   detections[i].y + detections[i].height / 2.0f);
            for (size_t j = 0; j < tracks_.size(); ++j) {
                cv::Point2f trk_center(tracks_[j]->current_pred.x + tracks_[j]->current_pred.width / 2.0f,
                                       tracks_[j]->current_pred.y + tracks_[j]->current_pred.height / 2.0f);
                float dist = cv::norm(det_center - trk_center);
                if (dist < max_dist_) {
                    pairs.push_back({(int)i, (int)j, dist});
                }
            }
        }

        std::sort(pairs.begin(), pairs.end(), [](const MatchPair& a, const MatchPair& b) { return a.dist < b.dist; });

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

        // 处理新目标，并打入专属的 Q 和 R 矩阵参数
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!det_matched[i]) {
                auto new_track = std::make_shared<Track>(); 
                new_track->kf.setNoiseParams(q_cov_, r_cov_); // 设置专属噪声参数
                new_track->kf.update(detections[i], true); 
                new_track->current_pred = detections[i];
                new_track->hit_streak = 1;
                new_track->id = next_id_++;
                tracks_.push_back(new_track);
            }
        }

        // 依据各轨道的专属寿命清理死亡目标
        int age_limit = max_age_;
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
            [age_limit](const std::shared_ptr<Track>& t) { return t->time_since_update > age_limit; }), tracks_.end());

        // 依据专属考核期进行渲染
        for (auto& t : tracks_) {
            if (t->time_since_update < 5 && t->hit_streak >= min_hits_) {
                cv::rectangle(frame, t->current_pred, color, 3);
                cv::putText(frame, label + ":" + std::to_string(t->id), 
                            cv::Point(t->current_pred.x, t->current_pred.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            }
            if (t->time_since_update > 0) t->hit_streak = 0; 
        }
    }

private:
    float q_cov_, r_cov_;
    int max_age_, min_hits_;
    float max_dist_;
    std::vector<std::shared_ptr<Track>> tracks_;
    int next_id_;
};


class PredictorNode : public rclcpp::Node {
public:
    PredictorNode() : Node("predictor_node"), 
        // 【核心调参区】：在这里为两个轨道注入不同的灵魂！
        // 传统视觉 (高噪声R=1.0, 较短寿命5帧, 严苛考核期4帧, 匹配距离100)
        tracker_trad_(1e-3, 1.0, 5, 4, 100.0f),
        // 神经网络 (低噪声R=1e-3, 超长记忆30帧, 极快确认期2帧, 匹配距离150)
        tracker_nn_(1e-2, 1e-3, 30, 2, 150.0f) 
    {
        rclcpp::QoS qos(1000); 
        sub_trad_img_ = this->create_subscription<sensor_msgs::msg::Image>("/traditional_vision/image_result", qos, std::bind(&PredictorNode::tradImgCb, this, _1));
        sub_trad_armors_ = this->create_subscription<rm_perception::msg::Armors>("/traditional_vision/armors", qos, std::bind(&PredictorNode::tradArmorsCb, this, _1));
        pub_trad_img_ = this->create_publisher<sensor_msgs::msg::Image>("/predictor/traditional_image", qos);
        
        sub_nn_img_ = this->create_subscription<sensor_msgs::msg::Image>("/neural_network/image_result", qos, std::bind(&PredictorNode::nnImgCb, this, _1));
        sub_nn_armors_ = this->create_subscription<rm_perception::msg::Armors>("/neural_network/armors", qos, std::bind(&PredictorNode::nnArmorsCb, this, _1));
        pub_nn_img_ = this->create_publisher<sensor_msgs::msg::Image>("/predictor/neural_image", qos);
        
        RCLCPP_INFO(this->get_logger(), "🎯 双轨独立参数预测节点启动！");
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
        for (const auto& a : armors_msg->armors) detections.emplace_back(a.corners[0].x, a.corners[0].y, a.corners[2].x - a.corners[0].x, a.corners[2].y - a.corners[0].y);
        
        // 传统视觉用黄色框 (CV) 标识
        tracker_trad_.processAndDraw(frame, detections, cv::Scalar(0, 255, 255), "CV");
        pub_trad_img_->publish(*cv_bridge::CvImage(img_msg->header, "bgr8", frame).toImageMsg());
    }

    void processNN() {
        if (nn_img_q_.empty() || nn_armors_q_.empty()) return;
        auto img_msg = nn_img_q_.front(); nn_img_q_.pop();
        auto armors_msg = nn_armors_q_.front(); nn_armors_q_.pop();
        cv::Mat frame = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
        std::vector<cv::Rect> detections;
        for (const auto& a : armors_msg->armors) detections.emplace_back(a.corners[0].x, a.corners[0].y, a.corners[2].x - a.corners[0].x, a.corners[2].y - a.corners[0].y);
        
        // 神经网络用亮绿色框 (NN) 标识
        tracker_nn_.processAndDraw(frame, detections, cv::Scalar(0, 255, 0), "NN");
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