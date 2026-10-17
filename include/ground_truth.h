#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 真值表读取与匹配。csv 列：image,armor_id,visible,tx,ty,tz，坐标为相机系、单位米。
// 匹配一律用三维距离而不是图像距离：透视只保留 x/z、y/z 的比值，图上位置接近的两块板
// 深度可能差很远，实测 image3 按图像距离会配错。
struct GroundTruth {
    std::string image;    // 所属图片名，如 image1
    int armor_id = 0;     // 装甲板编号 1~4
    bool visible = false; // true = 正面可见
    cv::Point3f t_cam;    // 相机系位置真值，单位米
};

// 读 csv。跳过注释行、空行、表头；剥掉 UTF-8 BOM 与行尾 \r；单行坏了只跳过该行。
std::vector<GroundTruth> loadGroundTruth(const std::string& csv_path);

// 筛出某张图里正面可见的装甲板（背面被挡住的本来就检测不到）
std::vector<GroundTruth> selectImage(const std::vector<GroundTruth>& all,
                                     const std::string& image_name);

// 找离解算结果最近的真值板（三维距离），超过 max_dist_mm 返回 nullptr 表示匹配不上
const GroundTruth* matchBy3DDistance(const cv::Vec3d& calc_mm,
                                     const std::vector<GroundTruth>& gts,
                                     double max_dist_mm,
                                     double* dist_out = nullptr);
