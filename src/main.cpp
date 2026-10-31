// 程序入口：解析命令行 → 分发到对应流水线。算法流程在 pipeline_*.cpp。
//   ./build/rm_vision_hw image [图片或目录] [真值csv] [内参yaml] [traditional|nn|both]
//   ./build/rm_vision_hw video [视频] [traditional|nn|both] [输出路径(可选)]
//   ./build/rm_vision_hw dump  [视频] [imu txt] [内参yaml] [方案] [输出csv] [外参编号(可选)]
//   ./build/rm_vision_hw ekf   [视频] [imu txt] [内参yaml] [方案] [输出视频] [输出csv]

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include "detectors.h"
#include "pipeline.h"

namespace {

void printUsage(const char* program) {
    std::cout
        << "用法:\n"
        << "  " << program << " image [图片或图片目录] [真值csv] [内参yaml] [traditional|nn|both]\n"
        << "  " << program << " video [视频] [traditional|nn|both] [输出路径(可选)]\n"
        << "  " << program << " dump [视频] [imu txt] [内参yaml] [方案] [输出csv] [外参编号(可选)]\n"
        << "  " << program << " ekf  [视频] [imu txt] [内参yaml] [方案] [输出视频] [输出csv]\n"
        << "  " << program << " plot [状态csv] [输出png]\n"
        << "\n"
        << "默认值:\n"
        << "  图片     data/images（目录下所有图片，每张各打印一张误差表）\n"
        << "  真值csv  data/gt/ground_truth.csv\n"
        << "  内参yaml 图片模式 config/camera_image.yaml，dump/ekf 模式 config/camera_video.yaml\n"
        << "  视频     data/videos/armor_record.avi\n"
        << "  imu txt  data/imu/imu_record.txt\n"
        << "  方案     traditional\n"
        << "  模型目录 models\n"
        << "  输出csv  dump: results/frame_poses.csv   ekf: results/ekf_states.csv\n"
        << "  输出视频 ekf: results/ekf_video.mp4\n"
        << "  plot      results/ekf_states.csv → results/ekf_curves.png\n"
        << "\n"
        << "示例:\n"
        << "  " << program << " image\n"
        << "  " << program << " image data/images/image2.jpg\n"
        << "  " << program << " image data/images data/gt/ground_truth_original.csv\n"
        << "  " << program << " video data/videos/armor_record.avi traditional\n"
        << "  " << program << " dump\n"
        << "  " << program << " ekf\n"
        << "  " << program << " plot\n"
        << "\n"
        << "dump 模式会逐帧解算装甲板位姿、配 IMU 姿态转到世界系，\n"
        << "并用\"目标车原地自转则四块板落在同一个圆上\"标定相机-IMU 外参。\n"
        << "ekf 模式在它之上跑扩展卡尔曼滤波，输出 9 个状态量的曲线数据和带重投影的视频。\n"
        << "plot 模式把那 9 个状态量画成曲线图。\n"
        << "\n"
        << "提示：要在项目根目录 rm_vision_hw/ 下运行。\n";
}

// 只看扩展名判断是不是要处理的图片，够用且不引入新依赖
bool isImageFile(const std::filesystem::path& p) {
    const std::string ext = p.extension().string();
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

// 把第二个参数展开成图片列表：传目录取其中所有图片并按名排序，传单文件就取它自己
std::vector<std::string> expandImageTargets(const std::string& target) {
    std::vector<std::string> out;
    std::error_code ec; // 用带 error_code 的重载，权限/路径问题不抛异常
    if (std::filesystem::is_directory(target, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(target, ec)) {
            if (std::filesystem::is_regular_file(entry.path(), ec) && isImageFile(entry.path())) {
                out.push_back(entry.path().string());
            }
        }
        std::sort(out.begin(), out.end()); // 目录遍历顺序不保证，排序后结果才可复现
    } else {
        out.push_back(target);
    }
    return out;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    const std::string mode = argv[1];
    const std::string model_dir = "models"; // 相对项目根目录，两个模式共用

    if (mode == "image") {
        const std::string target      = (argc > 2) ? argv[2] : "data/images";
        // 默认用 ground_truth.csv。材料的原始表 image 列整体错位一格，已按实测证据修正，
        // 未修正的那份存为 ground_truth_original.csv，可用来对比看错位的影响。
        const std::string gt_path     = (argc > 3) ? argv[3] : "data/gt/ground_truth.csv";
        const std::string config_path = (argc > 4) ? argv[4] : "config/camera_image.yaml";
        const std::string method_arg  = (argc > 5) ? argv[5] : "traditional";

        const std::vector<std::string> images = expandImageTargets(target);
        if (images.empty()) {
            std::cerr << "[错误] 没有找到要处理的图片: " << target << "\n"
                      << "       传单个图片文件路径，或者一个包含 .jpg/.png/.bmp 的目录。\n";
            return -1;
        }
        std::cout << "[信息] 待处理图片 " << images.size() << " 张: ";
        for (size_t i = 0; i < images.size(); ++i) {
            std::cout << (i ? ", " : "") << images[i];
        }
        std::cout << "\n";

        int ret = 0;
        for (const auto& image_path : images) {
            const int r = runImage(image_path, gt_path, config_path, method_arg, model_dir);
            if (r != 0) ret = r; // 某张失败不中断，继续处理剩下的
        }
        return ret;
    }

    if (mode == "video") {
        const std::string video_path  = (argc > 2) ? argv[2] : "data/videos/armor_record.avi";
        const std::string method_arg  = (argc > 3) ? argv[3] : "traditional";
        const std::string output_path = (argc > 4) ? argv[4] : std::string();
        return runVideo(video_path, method_arg, model_dir, output_path);
    }

    if (mode == "dump") {
        const std::string video_path  = (argc > 2) ? argv[2] : "data/videos/armor_record.avi";
        const std::string imu_path    = (argc > 3) ? argv[3] : "data/imu/imu_record.txt";
        // dump 要解算，必须用视频那套内参，默认不能再给图片那套
        const std::string config_path = (argc > 4) ? argv[4] : "config/camera_video.yaml";
        const std::string method_arg  = (argc > 5) ? argv[5] : "traditional";
        const std::string csv_path    = (argc > 6) ? argv[6] : "results/frame_poses.csv";
        // 最后一个参数一般留空让它自动挑外参；自动选错了才手动指定编号
        const std::string extrinsic   = (argc > 7) ? argv[7] : std::string();
        return runDump(video_path, imu_path, config_path, method_arg, model_dir, csv_path, extrinsic);
    }

    if (mode == "ekf") {
        const std::string video_path  = (argc > 2) ? argv[2] : "data/videos/armor_record.avi";
        const std::string imu_path    = (argc > 3) ? argv[3] : "data/imu/imu_record.txt";
        const std::string config_path = (argc > 4) ? argv[4] : "config/camera_video.yaml";
        const std::string method_arg  = (argc > 5) ? argv[5] : "traditional";
        const std::string video_out   = (argc > 6) ? argv[6] : "results/ekf_video.mp4";
        const std::string csv_out     = (argc > 7) ? argv[7] : "results/ekf_states.csv";
        return runEkf(video_path, imu_path, config_path, method_arg, model_dir, video_out, csv_out);
    }

    if (mode == "plot") {
        const std::string csv_path = (argc > 2) ? argv[2] : "results/ekf_states.csv";
        const std::string out_path = (argc > 3) ? argv[3] : "results/ekf_curves.png";
        return runPlot(csv_path, out_path);
    }

    std::cerr << "[错误] 不认识的模式: " << mode << "（只能是 image、video、dump、ekf 或 plot）\n\n";
    printUsage(argv[0]);
    return -1;
}
