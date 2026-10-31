#pragma once

#include "frame_pose.h"

#include <opencv2/opencv.hpp>
#include <array>

// 9 维 EKF：从每帧单块装甲板位置估计整车状态（状态量取自课件 slide 14 的 9 个）。
//   x = [c_x, v_x, c_y, v_y, z_a, v_z, theta, omega, r]
//   c 车中心(m)、v 速度(m/s)、z_a 板平均安装高度(m)、theta 车体朝向(rad)、
//   omega 角速度(rad/s)、r 板安装半径(m)
//
// 运动模型（匀速 CV）：
//   c' = c + v·dt,  z_a' = z_a + v_z·dt,  theta' = theta + omega·dt，其余视为常值
//
// 观测模型（第 j 块板的位置 = 车中心 + 随朝向转动的半径矢量）：
//   p_j = c + r_j·[cos(theta+phi_j), sin(theta+phi_j)]
//   z_j = z_a + (h_j - h_mean)
// phi_j 是板相对车头的方位角，r_j/h_j 是该板的半径与高度，各板差值与均值之差由已知几何补上。
// theta 用加号，与课件整体差 180°（课件定义的是"从板指向车中心"）。
class Ekf {
public:
    enum Idx { kCx = 0, kVx = 1, kCy = 2, kVy = 3, kZa = 4, kVz = 5, kTheta = 6, kOmega = 7, kR = 8 };
    static constexpr int kDim = 9;

    // 噪声参数：位置观测按坐标转换的实测残余给，角加速度给小值（目标转动平缓）
    struct Config {
        double q_accel_xy = 0.05;  // 水平加速度噪声谱密度 (m/s²)²·s
        double q_accel_z  = 0.15;  // 竖直方向不会有加速度，给小值
        double q_alpha    = 0.005; // 角加速度噪声 (rad/s²)²·s
        double q_omega    = 0.002; // omega 自身的随机游走
        double q_radius   = 1e-12; // 半径是官方几何，冻结不估
        double sigma_xy   = 0.080; // 位置观测噪声 (m)
        double sigma_z    = 0.020;
    };

    Ekf() = default;
    explicit Ekf(const Config& cfg) : cfg_(cfg) {}

    // 用启动阶段拟合出来的圆心、半径、高度和首帧解出的朝向初始化
    void init(double t, const cv::Vec2d& center, double radius, double theta, double height);
    bool initialized() const { return inited_; }

    // 预测到时刻 t，dt 由相邻 IMU 时间戳实算
    void predict(double t);

    // 按朝向连续性匹配板号：由观测反推 theta_b = atan2(p-c) - phi_b，取最接近当前 theta 的板，
    // 角度差超过 max_angle_rad 视为配不上（返回 -1）。不能用位置距离——单板可见时车中心与
    // 朝向耦合，位置误差会把结果带到隔壁板上。
    int matchByAngle(const cv::Vec3d& p, double max_angle_rad, double* angle_err = nullptr) const;

    // 按位置观测重定 theta（跳变处理用）
    void resetTheta(const cv::Vec3d& p);

    // 观测更新：board 是板号 0~3，p 是该板的世界系位置
    void update(int board, const cv::Vec3d& p);

    double get(int i) const { return x_.at<double>(i); }

    // 协方差接口：sigma(i) = sqrt(P_ii)，cov(i,j) = P_ij
    double sigma(int i) const;
    double cov(int i, int j) const;

    // 归一化新息平方 NIS = yᵀ S⁻¹ y，S = H P Hᵀ + R。观测 3 维，均值应约等于 3。
    // 远大于 3 说明协方差偏乐观（噪声给小了），远小于 3 说明偏保守。
    double lastNis() const { return last_nis_; }

    cv::Vec3d center() const;
    double theta() const { return x_.at<double>(kTheta); }
    double speed() const;

    // 第 board 块板在当前位置估计下的世界系中心
    cv::Vec3d armorCenter(int board) const;
    // 第 board 块板的四个角点（世界系），按 左上→右上→右下→左下 排，重投影画框用
    void armorCorners(int board, double width_m, double height_m,
                      std::array<cv::Point3d, 4>& out) const;

private:
    Config cfg_;
    bool inited_ = false;
    double last_t_ = 0.0;
    cv::Mat x_; // 9×1
    cv::Mat P_; // 9×9
    double last_nis_ = 0.0;
};
