#include "detectors.h"

#include <iostream>

namespace {

// 拼接 "models/xxx.onnx" 这类路径，目录末尾带不带 '/' 都行
std::string joinPath(const std::string& dir, const char* filename) {
    if (dir.empty()) return std::string(filename);
    if (dir.back() == '/' || dir.back() == '\\') return dir + filename;
    return dir + "/" + filename;
}

} // namespace

bool parseMethods(const std::string& arg, std::vector<std::string>& out) {
    out.clear();
    if (arg == method::kTraditional) {
        out.push_back(method::kTraditional);
    } else if (arg == method::kNeural) {
        out.push_back(method::kNeural);
    } else if (arg == method::kBoth) {
        out.push_back(method::kTraditional);
        out.push_back(method::kNeural);
    } else {
        std::cerr << "[错误] 不认识的方案名: " << arg
                  << "（只能是 traditional / nn / both）\n";
        return false;
    }
    return true;
}

bool DetectorSet::init(const std::string& model_dir,
                       const std::vector<std::string>& methods) {
    bool need_digit = false; // 传统方案要拿数字网络做配对校验
    bool need_yolo = false;

    for (const auto& m : methods) {
        if (m == method::kTraditional) need_digit = true;
        if (m == method::kNeural) need_yolo = true;
    }

    bool all_ok = true;

    if (need_digit) {
        const std::string onnx = joinPath(model_dir, "digit.onnx");
        const std::string label = joinPath(model_dir, "label.txt");
        if (!digit_.load(onnx, label)) {
            // 模型缺失只降级不退出，否则一个没下载的模型会让整条链路都跑不了
            std::cerr << "[提示] 数字识别模型没加载成功，传统方案将退化为"
                         "「只做几何配对，不校验数字」。\n"
                         "       检测结果仍然可用，只是画面上不会显示数字。\n"
                         "       需要模型时请看 models/README.md 里的下载命令。\n";
            all_ok = false;
        }
    }

    if (need_yolo) {
        const std::string onnx = joinPath(model_dir, "armor.onnx");
        if (!yolo_.load(onnx)) {
            std::cerr << "[提示] 装甲板检测模型没加载成功，神经网络方案不可用。\n"
                         "       需要模型时请看 models/README.md 里的下载命令。\n";
            all_ok = false;
        }
    }

    return all_ok;
}

std::vector<ArmorDetection> DetectorSet::detect(const cv::Mat& frame,
                                                const std::string& method_name,
                                                cv::Mat* debug_mask) {
    if (method_name == method::kNeural) {
        if (!yolo_.loaded()) return {}; // init 里已警告过，这里安静返回空
        // 类名和数字由 YOLODetector 自己填，它才知道检测网络这 9 类的含义；
        // 这里用 digit_.labelName() 去翻译会全错。
        return yolo_.detect(frame);
    }

    std::vector<ArmorDetection> dets = traditional_.detect(frame, debug_mask);

    if (!digit_.loaded()) return dets; // 没有数字模型就原样返回

    // 数字网络做最后一道校验。原则：有把握地说没有数字才删，只是没把握不删。
    std::vector<ArmorDetection> kept;
    kept.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        auto& d = dets[i];

        float conf = 0.f;
        const int class_id = digit_.classifyKeepNegative(frame, d, &conf);
        const bool sure = (conf >= digit_.threshold());

        enum Verdict { kKeepWithDigit, kDropNegative, kKeepUnknown };
        Verdict verdict;
        if (class_id < 0) {
            verdict = kKeepUnknown;   // 框退化成一个点拉不正，或推理没跑起来
        } else if (digit_.isNegative(class_id) && sure) {
            verdict = kDropNegative;  // 网络明确说该区域内没有数字
        } else if (sure) {
            verdict = kKeepWithDigit; // 认出来了，把数字写上去
        } else {
            verdict = kKeepUnknown;   // 有类别但没把握：板保留，不写数字
        }

        if (verbose_) {
            std::cout << "[信息] 数字校验 板#" << (i + 1) << ": ";
            if (verdict == kDropNegative) {
                std::cout << "判为负样本（" << static_cast<int>(conf * 100)
                          << "%），丢弃\n";
            } else if (verdict == kKeepWithDigit) {
                std::cout << "认出 " << digit_.labelName(class_id)
                          << "（" << static_cast<int>(conf * 100)
                          << "%），保留\n";
            } else {
                std::cout << "没把握，保留但不标注数字\n";
            }
        }

        if (verdict == kDropNegative) continue;

        if (verdict == kKeepWithDigit) {
            d.class_id = class_id;
            d.class_name = digit_.labelName(class_id);
            d.digit = DigitClassifier::digitOf(class_id);
            d.confidence = conf;
        }
        kept.push_back(d);
    }

    // 全部被判负就输出空，不再退回纯几何结果：那条兜底会把队标、窗框、人手都放回来，
    // 而它想救的"真板被误判 negative"已由扶正 ROI 复核解决。
    if (kept.empty() && !dets.empty() && verbose_) {
        std::cout << "[提示] " << dets.size()
                  << " 块候选全被判为负样本（候选区域里没有数字），本帧无输出。\n";
    }
    return kept;
}
