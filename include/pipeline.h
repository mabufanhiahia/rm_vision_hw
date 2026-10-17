#pragma once

#include <string>

// 两条流水线：图片模式与视频模式。main.cpp 只负责解析命令行并分发到这里。

// 图片模式：读内参 → 读真值 → 读图 → 识别 → PnP → 比对真值 → 打印误差表 → 画图存图
int runImage(const std::string& image_path, const std::string& gt_path,
             const std::string& config_path, const std::string& method_arg,
             const std::string& model_dir);

// 视频模式：逐帧读 → 识别 → 画框与数字 → 写视频 → 实时预览（作业不要求视频解算，故不读内参）
int runVideo(const std::string& video_path, const std::string& method_arg,
             const std::string& model_dir, const std::string& output_path);
