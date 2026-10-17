#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// 数据源：内参读取、图片与视频读写。算法代码只管吃 cv::Mat，不关心来源。
//
// 本项目两套内参严格分开，拿错不会报错，只会安静地给出偏离很大的结果：
//   config/camera_image.yaml  图片 960×720，畸变全 0
//   config/camera_video.yaml  视频 1440×1080，带畸变

struct CameraIntrinsics {
    cv::Mat camera_matrix; // 3×3 内参，CV_64F
    cv::Mat dist_coeffs;   // 畸变系数 1×5（k1,k2,p1,p2,k3），全 0 表示无畸变
    cv::Size image_size;   // 标定分辨率，仅用于提示核对，不参与解算
    bool valid = false;
};

namespace camera {

// 读内参。失败返回 valid=false，由调用方决定报错退出还是继续
CameraIntrinsics loadIntrinsics(const std::string& yaml_path);

// 读图，失败返回空 Mat。单独包一层是为了报错时能说清是哪个路径读不到
cv::Mat loadImage(const std::string& path);

// 打开视频。fps 读不到按 30 兜底，total 读不到为 0 属正常
bool openVideo(const std::string& path, cv::VideoCapture& cap,
               double* fps = nullptr, int* total = nullptr);

// 打开视频写出器。size 必须与实际写入帧一致，否则文件打不开
bool openVideoWriter(const std::string& path, cv::VideoWriter& writer,
                     double fps, cv::Size size);

} // namespace camera
