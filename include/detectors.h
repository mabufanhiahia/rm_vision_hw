#pragma once

#include "digit_classifier.h"
#include "traditional_detector.h"
#include "types.h"
#include "YOLO_detector.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 两种识别方案的统一入口，图片模式和视频模式共用同一套模型加载逻辑。
//   "traditional"  传统视觉（几何配对 + 数字网络校验）
//   "nn"           神经网络
//   "both"         两种都跑
namespace method {
inline const char* kTraditional = "traditional";
inline const char* kNeural = "nn";
inline const char* kBoth = "both";
} // namespace method

// 解析方案名，"both" 展开成两项，不认识的返回 false
bool parseMethods(const std::string& arg, std::vector<std::string>& out);

// 识别器集合。传统方案在几何配对之后追加一道数字网络校验：
// 两灯条的几何关系可能像一对，但凑出来的区域里根本没有数字，据此把跨板配对剔掉。
class DetectorSet {
public:
    // 按需加载模型。模型缺失只告警不退出，让对应方案降级，不影响另一条路径。
    bool init(const std::string& model_dir, const std::vector<std::string>& methods);

    // debug_mask 只有传统方案用（颜色分割掩膜），神经网络方案忽略
    std::vector<ArmorDetection> detect(const cv::Mat& frame, const std::string& method,
                                       cv::Mat* debug_mask = nullptr);

    bool nnReady() const { return yolo_.loaded(); }
    bool digitReady() const { return digit_.loaded(); }

    // 是否打印每块板的数字校验判定。视频每帧都调 detect()，只有图片模式才打开。
    void setVerbose(bool v) { verbose_ = v; }

private:
    TraditionalDetector traditional_;
    DigitClassifier digit_;
    YOLODetector yolo_;
    bool verbose_ = false;
};
