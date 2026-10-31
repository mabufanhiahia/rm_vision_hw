#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 一行 IMU 内录数据。文件里每行的顺序是：时间戳 w x y z。
struct ImuSample {
    double timestamp = 0.0;    // 文件里写的时间戳，这是"收到时刻"，不是采集时刻
    double capture_time = 0.0; // 真正的采集时刻 = timestamp - offset
    cv::Vec4d quat;            // 四元数，按文件顺序存成 (w, x, y, z)
};

class ImuReader {
public:
    // 读 IMU 文件。offset_s 是"收到时刻"退到采集时刻的偏移（9ms），四元数读入后归一化。
    // 第 k 帧配第 k 行，所以不按时间重排，保持文件顺序。
    static bool load(const std::string& path, double offset_s,
                     std::vector<ImuSample>& out, std::string* err = nullptr);

    // 四元数 (w,x,y,z) → 3×3 旋转矩阵
    static cv::Mat quatToRot(const cv::Vec4d& q);

    // 旋转矩阵 → 欧拉角（弧度），顺序为 X(roll) Y(pitch) Z(yaw)
    static cv::Vec3d rotToEuler(const cv::Mat& R);
};
