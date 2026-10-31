#pragma once

#include <string>

// 五条流水线：图片、视频、dump、ekf、plot。main.cpp 只负责解析命令行并分发到这里。

// 图片模式：读内参 → 读真值 → 读图 → 识别 → PnP → 比对真值 → 打印误差表 → 画图存图
int runImage(const std::string& image_path, const std::string& gt_path,
             const std::string& config_path, const std::string& method_arg,
             const std::string& model_dir);

// 视频模式：逐帧读 → 识别 → 画框与数字 → 写视频 → 实时预览（作业不要求视频解算，故不读内参）
int runVideo(const std::string& video_path, const std::string& method_arg,
             const std::string& model_dir, const std::string& output_path);

// dump 模式：逐帧识别 → PnP → 配 IMU 姿态转到世界系，顺带标定相机-IMU 外参，导出位姿 CSV。
// extrinsic_arg 为空时自动挑候选，也可以直接传候选编号。
int runDump(const std::string& video_path, const std::string& imu_path,
            const std::string& config_path, const std::string& method_arg,
            const std::string& model_dir, const std::string& csv_path,
            const std::string& extrinsic_arg);

// ekf 模式：在 dump 的数据链路上加 EKF，估计整车 9 维状态。第一遍取位姿并拟合圆心
// 当初值，第二遍逐帧滤波、画四块板重投影与状态面板，导出状态序列 CSV。
int runEkf(const std::string& video_path, const std::string& imu_path,
           const std::string& config_path, const std::string& method_arg,
           const std::string& model_dir, const std::string& video_out,
           const std::string& csv_out);

// plot 模式：读 ekf 导出的状态序列，把 9 个状态量画成 3×3 的曲线图。
// 用 OpenCV 画，不引入 matplotlib，编译完就能出图。
int runPlot(const std::string& csv_path, const std::string& out_path);
