#include "rm_perception/traditional_detector.hpp"

namespace rm_perception {

std::vector<TraditionalArmor> TraditionalDetector::detect(const cv::Mat& src, int enemy_color) {
    std::vector<TraditionalArmor> armors;
    if (src.empty()) return armors;

    bin_image_ = preprocess(src, enemy_color);
    std::vector<LightBar> lights = findLights(bin_image_);
    armors = matchLights(lights, enemy_color, bin_image_);

    return armors;
}

cv::Mat TraditionalDetector::preprocess(const cv::Mat& src, int enemy_color) {
    cv::Mat final_bin_img;
    std::vector<cv::Mat> channels;
    cv::split(src, channels); 

    if (enemy_color == 1) {
        // ==========================================
        // 【红方阵营】：硬编码的双阈值防断裂逻辑
        // ==========================================
        const int red_color_thresh = 50; 
        const int red_gray_thresh = 43;  

        cv::Mat gray_img, color_mask, gray_mask;
        cv::cvtColor(src, gray_img, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_img, gray_mask, red_gray_thresh, 255, cv::THRESH_BINARY);

        cv::subtract(channels[2], channels[0], color_mask); // R - B
        cv::threshold(color_mask, color_mask, red_color_thresh, 255, cv::THRESH_BINARY);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(color_mask, color_mask, kernel);
        cv::bitwise_and(color_mask, gray_mask, final_bin_img);
        cv::morphologyEx(final_bin_img, final_bin_img, cv::MORPH_CLOSE, kernel);

    } else {
        // ==========================================
        // 【蓝方阵营】：硬编码的极简通道相减逻辑
        // ==========================================
        const int blue_color_thresh = 80; 
        
        cv::Mat color_mask;
        cv::subtract(channels[0], channels[2], color_mask); // B - R
        cv::threshold(color_mask, final_bin_img, blue_color_thresh, 255, cv::THRESH_BINARY);
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::dilate(final_bin_img, final_bin_img, kernel);
    }

    return final_bin_img;
}

std::vector<LightBar> TraditionalDetector::findLights(const cv::Mat& bin_img) {
    std::vector<LightBar> lights;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        if (cv::contourArea(contour) < min_light_area_) continue;

        cv::RotatedRect rrect = cv::minAreaRect(contour);
        double width = rrect.size.width;
        double height = rrect.size.height;
        double angle = rrect.angle;

        if (width > height) {
            std::swap(width, height);
            angle += 90.0;
        }
        if (angle > 90.0) angle -= 180.0;
        if (angle < -90.0) angle += 180.0;

        // 远距离灯条长宽比退化特批
        bool is_valid_ratio = (height / width > 1.2) || (height < 15 && height / width > 1.0);

        if (is_valid_ratio && std::abs(angle) < max_light_angle_) {
            LightBar light;
            light.rect = rrect;
            light.length = height;
            light.angle = angle;
            light.center = rrect.center;
            lights.push_back(light);
        }
    }
    return lights;
}

std::vector<TraditionalArmor> TraditionalDetector::matchLights(const std::vector<LightBar>& lights, int enemy_color, const cv::Mat& bin_img) {
    std::vector<TraditionalArmor> armors;
    if (lights.size() < 2) return armors;

    std::vector<LightBar> sorted_lights = lights;
    std::sort(sorted_lights.begin(), sorted_lights.end(), 
              [](const LightBar& l1, const LightBar& l2) { return l1.center.x < l2.center.x; });

    std::vector<bool> used(sorted_lights.size(), false);

    for (size_t i = 0; i < sorted_lights.size() - 1; ++i) {
        if (used[i]) continue; 

        for (size_t j = i + 1; j < sorted_lights.size(); ++j) {
            if (used[j]) continue; 

            const LightBar& l1 = sorted_lights[i];
            const LightBar& l2 = sorted_lights[j];

            if (std::abs(l1.angle - l2.angle) > max_angle_diff_) continue;

            double length_ratio = std::max(l1.length, l2.length) / std::min(l1.length, l2.length);
            if (length_ratio > 1.8) continue; 

            double distance = cv::norm(l1.center - l2.center);
            double avg_length = (l1.length + l2.length) / 2.0;
            double armor_ratio = distance / avg_length;
            if (armor_ratio < min_armor_ratio_ || armor_ratio > max_armor_ratio_) continue;

            double y_diff = std::abs(l1.center.y - l2.center.y);
            double x_diff = std::abs(l1.center.x - l2.center.x);
            double armor_tilt_angle = std::atan2(y_diff, x_diff) * 180.0 / CV_PI;
            if (armor_tilt_angle > 20.0) continue; 

            cv::Point2f center = (l1.center + l2.center) / 2.0;

            // 中心黑洞校验 (防实体反光误检)
            int roi_w = std::max(1, (int)(distance * 0.33));
            int roi_h = std::max(1, (int)(avg_length * 0.33));
            cv::Rect center_roi(center.x - roi_w / 2, center.y - roi_h / 2, roi_w, roi_h);

            center_roi.x = std::max(0, center_roi.x);
            center_roi.y = std::max(0, center_roi.y);
            if (center_roi.x + center_roi.width > bin_img.cols) center_roi.width = bin_img.cols - center_roi.x;
            if (center_roi.y + center_roi.height > bin_img.rows) center_roi.height = bin_img.rows - center_roi.y;

            cv::Mat roi_img = bin_img(center_roi);
            int white_pixels = cv::countNonZero(roi_img);
            double white_ratio = (double)white_pixels / (center_roi.width * center_roi.height);

            if (white_ratio > 0.15) continue; 

            TraditionalArmor armor;
            armor.color = enemy_color;
            armor.center = center;

            cv::Point2f left_pts[4], right_pts[4];
            l1.rect.points(left_pts); 
            l2.rect.points(right_pts); 

            std::vector<cv::Point2f> left_sort(left_pts, left_pts + 4);
            std::vector<cv::Point2f> right_sort(right_pts, right_pts + 4);
            std::sort(left_sort.begin(), left_sort.end(), [](const cv::Point2f& p1, const cv::Point2f& p2){ return p1.y < p2.y; });
            std::sort(right_sort.begin(), right_sort.end(), [](const cv::Point2f& p1, const cv::Point2f& p2){ return p1.y < p2.y; });

            armor.corners[0] = cv::Point2f((left_sort[0].x + left_sort[1].x)/2, (left_sort[0].y + left_sort[1].y)/2); 
            armor.corners[1] = cv::Point2f((left_sort[2].x + left_sort[3].x)/2, (left_sort[2].y + left_sort[3].y)/2); 
            armor.corners[3] = cv::Point2f((right_sort[0].x + right_sort[1].x)/2, (right_sort[0].y + right_sort[1].y)/2); 
            armor.corners[2] = cv::Point2f((right_sort[2].x + right_sort[3].x)/2, (right_sort[2].y + right_sort[3].y)/2); 

            armors.push_back(armor);
            used[i] = true;
            used[j] = true;
            break; 
        }
    }
    return armors;
}

} // namespace rm_perception