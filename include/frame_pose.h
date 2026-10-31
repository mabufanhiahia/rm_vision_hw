#pragma once

#include "imu_reader.h"
#include "types.h"

#include <array>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 装甲板几何，取自标注文件 1.2 节 odom_vision 世界系的 /tf 真值，三份交叉验证一致。
//   phi 车体系方位角（armor_2 为车头方向）、r 板中心到车中心距离、z 板中心高度
// 两种安装高度差 14 mm（左右 0.2402、前后 0.2262），两组半径只差 0.6 mm。
// 下标对应标注文件里的 armor_1..4。
namespace armor_geom {
constexpr int kCount = 4;
constexpr double kPhiDeg[kCount]   = {90.0,   0.0, 270.0, 180.0};
constexpr double kRadiusM[kCount]  = {0.208584, 0.209049, 0.208337, 0.209070};
constexpr double kHeightM[kCount]  = {0.239977, 0.225978, 0.240411, 0.226411};
constexpr double kRadiusMeanM = 0.20876; // 四块平均，标定外参时当已知几何用
} // namespace armor_geom

// 一帧里一块装甲板解出来的位姿。位置统一用米（PnP 出来是毫米，进这里之前先除 1000）。
struct ArmorPoseCam {
    int frame = 0;
    int digit = -1;                            // 数字识别结果，配板时当 ID 用
    cv::Vec3d pos = cv::Vec3d(0, 0, 0);        // 相机系位置
    cv::Vec3d rvec = cv::Vec3d(0, 0, 0);       // 相机系朝向（旋转向量）
    double reproj = -1.0;                      // 重投影误差(px)
    std::array<cv::Point2f, 4> image_points;   // 原始角点，重投影画框时要对照
};

// 相机相对 IMU 的安装旋转。R_ic 满足 P_I = R_ic·P_C；
// transpose_imu 表示四元数是反约定，用前要先转置。
struct ExtrinsicOption {
    std::string name;
    cv::Mat R_ic;
    bool transpose_imu = false;
};

// 穷举轴对齐安装：3 轴 6 种排列 × 右手系符号 = 24 个，再乘四元数是否转置，共 48 个
std::vector<ExtrinsicOption> enumerateExtrinsics();

// 由 IMU 采样和候选外参算出"相机系 → 世界系"的旋转 R_wc
cv::Mat worldRotation(const cv::Vec4d& quat, const ExtrinsicOption& opt);

// 相机系点 → 世界系点。世界系原点取在相机光心，只旋转不平移（没有里程计）
cv::Vec3d camToWorld(const cv::Vec3d& p_cam, const cv::Mat& R_wc);
