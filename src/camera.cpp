#include "camera.h"

#include <cctype>
#include <cmath>
#include <iostream>

CameraIntrinsics camera::loadIntrinsics(const std::string& yaml_path) {
    CameraIntrinsics cam;

    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[错误] 打不开相机配置文件: " << yaml_path << "\n"
                  << "       提示：要在项目根目录 rm_vision_hw/ 下运行，或传入正确路径。\n";
        return cam; // valid 保持 false
    }

    fs["camera_matrix"] >> cam.camera_matrix;
    fs["distortion_coefficients"] >> cam.dist_coeffs;

    // 分辨率是选读项，读不到留 0，不影响解算
    int w = 0, h = 0;
    fs["image_width"] >> w;
    fs["image_height"] >> h;
    cam.image_size = cv::Size(w, h);

    fs.release();

    if (cam.camera_matrix.empty() || cam.dist_coeffs.empty()) {
        std::cerr << "[错误] 配置里缺少 camera_matrix 或 distortion_coefficients: "
                  << yaml_path << "\n";
        return cam;
    }
    if (cam.camera_matrix.rows != 3 || cam.camera_matrix.cols != 3) {
        std::cerr << "[错误] camera_matrix 必须是 3×3，实际是 "
                  << cam.camera_matrix.rows << "×" << cam.camera_matrix.cols << "\n";
        return cam;
    }

    cam.valid = true;
    return cam;
}

cv::Mat camera::loadImage(const std::string& path) {
    const cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "[错误] 读不到图片: " << path << "\n"
                  << "       常见原因：1) 不在项目根目录运行，相对路径找不到；\n"
                  << "                2) 文件名大小写不对（Linux 区分大小写）；\n"
                  << "                3) 文件损坏或格式不被 OpenCV 支持。\n";
    }
    return img;
}

bool camera::openVideo(const std::string& path, cv::VideoCapture& cap,
                       double* fps, int* total) {
    if (!cap.open(path)) {
        std::cerr << "[错误] 打不开视频: " << path << "\n"
                  << "       提示：.avi 需要系统里有对应的解码器（ffmpeg）。\n"
                  << "       可用 ffmpeg -i " << path << " 先确认文件本身没问题。\n";
        return false;
    }
    if (fps != nullptr) {
        *fps = cap.get(cv::CAP_PROP_FPS);
        if (*fps <= 0.0 || std::isnan(*fps)) *fps = 30.0; // 读不到按 30 兜底
    }
    if (total != nullptr) {
        *total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    }
    return true;
}

bool camera::openVideoWriter(const std::string& path, cv::VideoWriter& writer,
                             double fps, cv::Size size) {
    // 按扩展名挑编码器：.mp4 用 MP4V，其余（本项目默认 .avi）用 XVID
    std::string lower = path;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const bool is_mp4 = lower.size() >= 4 && lower.compare(lower.size() - 4, 4, ".mp4") == 0;

    const int mp4v = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
    const int xvid = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    const int preferred = is_mp4 ? mp4v : xvid;

    if (writer.open(path, preferred, fps, size, true)) return true;

    // 首选编码器没编译进去时换另一个再试
    std::cerr << "[警告] 首选编码器不可用，改用备用编码器\n";
    return writer.open(path, is_mp4 ? xvid : mp4v, fps, size, true);
}
