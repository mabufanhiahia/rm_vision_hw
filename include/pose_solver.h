#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>
#include <vector>

// PnP 位置解算：输入装甲板 4 个图像角点，输出它在相机坐标系下的位置与朝向。
class PoseSolver {
public:
    static constexpr double kMaxReprojError = 3.0; // 重投影误差上限(px)，超过认为解算不可信

    // 装甲板物理尺寸(mm)。宽度必须是两灯条中心距而非外缘宽度，改错会带来固定比例的 z 偏差。
    static constexpr float kDefaultArmorWidthMm = 135.0f;
    static constexpr float kDefaultArmorHeightMm = 55.0f;

    // 图片与视频是两套内参，不能混用
    PoseSolver(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
               float armor_width, float armor_height);

    PoseResult solve(const ArmorDetection& detection);

private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    std::vector<cv::Point3f> object_points_; // 装甲板四角物理坐标，物体坐标系，单位 mm
};
