#include "imu_reader.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

bool ImuReader::load(const std::string& path, double offset_s,
                     std::vector<ImuSample>& out, std::string* err) {
    std::ifstream in(path);
    if (!in) {
        if (err) *err = "打不开 IMU 文件: " + path;
        return false;
    }

    out.clear();
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) continue;

        std::istringstream iss(line);
        double t, w, x, y, z;
        if (!(iss >> t >> w >> x >> y >> z)) {
            // 文件开头可能有表头，或者是空行，跳过不致命
            std::cerr << "[提示] IMU 第 " << line_no << " 行格式不对，已跳过: " << line << "\n";
            continue;
        }

        ImuSample s;
        s.timestamp = t;
        s.capture_time = t - offset_s;
        s.quat = cv::Vec4d(w, x, y, z);

        const double n = std::sqrt(w * w + x * x + y * y + z * z);
        if (n < 1e-9) {
            if (err) *err = "IMU 第 " + std::to_string(line_no) + " 行四元数模长为 0";
            return false;
        }
        s.quat /= n; // 归一化，后面转矩阵才严格正交
        out.push_back(s);
    }

    if (out.empty()) {
        if (err) *err = "IMU 文件里没有读到有效数据: " + path;
        return false;
    }
    return true;
}

cv::Mat ImuReader::quatToRot(const cv::Vec4d& q) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];

    // 四元数转旋转矩阵的标准展开式
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    R.at<double>(0, 0) = 1 - 2 * (y * y + z * z);
    R.at<double>(0, 1) = 2 * (x * y - w * z);
    R.at<double>(0, 2) = 2 * (x * z + w * y);
    R.at<double>(1, 0) = 2 * (x * y + w * z);
    R.at<double>(1, 1) = 1 - 2 * (x * x + z * z);
    R.at<double>(1, 2) = 2 * (y * z - w * x);
    R.at<double>(2, 0) = 2 * (x * z - w * y);
    R.at<double>(2, 1) = 2 * (y * z + w * x);
    R.at<double>(2, 2) = 1 - 2 * (x * x + y * y);
    return R;
}

cv::Vec3d ImuReader::rotToEuler(const cv::Mat& R) {
    // atan2 的写法在 pitch 接近 ±90° 时才退化，本场景俯仰只有十几度，够用
    const double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                                R.at<double>(1, 0) * R.at<double>(1, 0));
    cv::Vec3d e;
    if (sy > 1e-6) {
        e[0] = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2)); // roll  X
        e[1] = std::atan2(-R.at<double>(2, 0), sy);                // pitch Y
        e[2] = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0)); // yaw   Z
    } else {
        e[0] = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        e[1] = std::atan2(-R.at<double>(2, 0), sy);
        e[2] = 0.0;
    }
    return e;
}
