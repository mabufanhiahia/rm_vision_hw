#include "pipeline.h"

#include "camera.h"
#include "detectors.h"
#include "visualizer.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

// 每处理这么多帧打印一次进度，避免刷屏
constexpr int kProgressEvery = 30;

struct Channel {
    std::string method;      // traditional / nn
    cv::VideoWriter writer;  // 对应的输出视频
    std::string out_path;
    long long total_detected = 0; // 累计检出装甲板数
};

std::string toFixed1(double v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << v;
    return ss.str();
}

std::string toFixed0(double v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(0) << v;
    return ss.str();
}

cv::Scalar colorOf(const std::string& m) {
    return (m == method::kNeural) ? cv::Scalar(255, 255, 0)  // 青色 (BGR)
                                  : cv::Scalar(0, 255, 0);   // 绿色
}

} // namespace

int runVideo(const std::string& video_path, const std::string& method_arg,
             const std::string& model_dir, const std::string& output_path) {
    std::vector<std::string> methods;
    if (!parseMethods(method_arg, methods)) return -1;

    cv::VideoCapture cap;
    double fps = 30.0;
    int total_frames = 0;
    if (!camera::openVideo(video_path, cap, &fps, &total_frames)) return -1;

    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "========== 视频模式 ==========\n"
              << "输入: " << video_path << "  " << width << "x" << height
              << "  " << toFixed1(fps) << " fps";
    if (total_frames > 0) std::cout << "  共 " << total_frames << " 帧";
    std::cout << "\n方案: " << method_arg << "\n";

    DetectorSet detectors;
    if (!detectors.init(model_dir, methods)) {
        std::cerr << "[警告] 部分模型未加载，对应方案可能检不出东西。\n";
    }

    std::filesystem::create_directories("results");
    std::vector<Channel> channels;
    for (const auto& m : methods) {
        channels.emplace_back(); // 直接在 vector 里构造，避免拷贝/移动
        Channel& ch = channels.back();
        ch.method = m;
        // 指定了输出路径就直接用；跑两种方案时必须各写一个文件，按方案名区分
        ch.out_path = (output_path.empty() || methods.size() > 1)
            ? ("results/video_" + m + ".avi")
            : output_path;
        if (!camera::openVideoWriter(ch.out_path, ch.writer, fps, cv::Size(width, height))) {
            std::cerr << "[错误] 打不开输出视频: " << ch.out_path << "\n";
            return -1;
        }
    }
    for (const auto& ch : channels) {
        std::cout << "输出: " << ch.out_path << "\n";
    }
    std::cout << "处理中...（按 q 或 ESC 可以提前结束）\n";

    const auto t_start = std::chrono::steady_clock::now();
    cv::Mat frame;
    int frame_index = 0;
    int processed = 0;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;
        ++frame_index;

        cv::Mat preview; // 第一个通道的画面拿来 imshow
        bool first = true;

        for (auto& ch : channels) {
            cv::Mat canvas = frame.clone();

            const auto t0 = std::chrono::steady_clock::now();
            // 只有传统方案要 debug_mask，视频里不用，传 nullptr
            const std::vector<ArmorDetection> dets = detectors.detect(canvas, ch.method);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            ch.total_detected += static_cast<long long>(dets.size());

            for (const auto& d : dets) {
                Visualizer::drawArmor(canvas, d, colorOf(ch.method));
            }

            // 左上角信息条：帧号 / 方案 / 本帧检出数 / 本帧耗时，调参时能直接看出从哪帧开始丢目标
            Visualizer::drawInfoBar(canvas, {
                "frame: " + std::to_string(frame_index) +
                    (total_frames > 0 ? "/" + std::to_string(total_frames) : ""),
                "method: " + ch.method,
                "detected: " + std::to_string(dets.size()),
                "infer: " + toFixed1(ms) + " ms"
            });

            ch.writer.write(canvas);
            if (first) { preview = canvas; first = false; }
        }

        ++processed;

        // 1440×1080 的帧必须缩到屏幕装得下，否则窗口被顶到屏幕外
        if (!preview.empty()) {
            Visualizer::show("video_result [" + method_arg + "]", preview, 960, 600);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 'Q' || key == 27) { // 27 = ESC
                std::cout << "[提示] 用户提前结束处理（第 " << frame_index << " 帧）\n";
                break;
            }
        }

        if (frame_index % kProgressEvery == 0) {
            std::cout << "  已处理 " << frame_index << " 帧" << std::endl;
        }
    }

    cv::destroyAllWindows();
    for (auto& ch : channels) ch.writer.release();
    cap.release();

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "========== 汇总 ==========\n"
              << "处理帧数: " << processed << "\n"
              << "总耗时: " << toFixed1(elapsed_s) << " s";
    if (elapsed_s > 0.0) {
        std::cout << "（平均 " << toFixed1(processed / elapsed_s) << " fps）";
    }
    std::cout << "\n";
    for (const auto& ch : channels) {
        std::cout << "  " << ch.method << ": 累计检出 " << ch.total_detected
                  << " 块，平均每帧 "
                  << (processed > 0 ? toFixed1(static_cast<double>(ch.total_detected) / processed)
                                    : std::string("0.0"))
                  << " 块 → " << ch.out_path << "\n";
    }
    std::cout << "==========================\n";
    return 0;
}
