// 程序入口：解析命令行 → 分发到对应流水线。算法流程在 pipeline_image.cpp / pipeline_video.cpp。
//
//   ./build/rm_vision_hw image [图片或目录] [真值csv] [内参yaml] [traditional|nn|both]
//   ./build/rm_vision_hw video [视频] [traditional|nn|both] [输出路径(可选)]
//
// 图片与视频的内参是两套，不要混用（图片 960×720 无畸变，视频 1440×1080 带畸变）。

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
        << "\n"
        << "默认值:\n"
        << "  图片     data/images（目录下所有图片，每张各打印一张误差表）\n"
        << "  真值csv  data/gt/ground_truth_reordered.csv\n"
        << "  内参yaml config/camera_image.yaml\n"
        << "  方案     traditional\n"
        << "  模型目录 models\n"
        << "\n"
        << "示例:\n"
        << "  " << program << " image\n"
        << "  " << program << " image data/images/image2.jpg\n"
        << "  " << program << " image data/images data/gt/ground_truth_reordered.csv\n"
        << "  " << program << " video data/videos/armor_record.avi traditional\n"
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
        // 默认用重排后的真值表：原表的 image 列与图片编号错位一格，直接用会匹配不上
        const std::string gt_path     = (argc > 3) ? argv[3] : "data/gt/ground_truth_reordered.csv";
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

    std::cerr << "[错误] 不认识的模式: " << mode << "（只能是 image 或 video）\n\n";
    printUsage(argv[0]);
    return -1;
}
