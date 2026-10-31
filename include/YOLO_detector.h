#pragma once

#include "types.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// 神经网络装甲板检测：ONNX 模型直接输出四个关键点 + 兵种类别 + 置信度。
// 输出布局按通道数分两种：
//   V8-21（1×21×6300）：[0..3] 4 色得分 → [4..12] 9 类 → [13..20] 4 个关键点(x,y)
//   V5-22（1×22×N）   ：[0..7] 4 个关键点 → [8] 置信度 logit → [9..12] 4 色 → [13..21] 9 类
class YOLODetector {
public:
    static constexpr float kDefaultConfidence = 0.50f; // 置信度低于它直接丢掉
    static constexpr float kDefaultNms = 0.45f;        // 相邻两块板常并排，阈值放宽避免互相压掉

    ~YOLODetector();

    bool load(const std::string& onnx_path); // 失败返回 false，不抛异常

    bool loaded() const { return loaded_; }

    void setConfidenceThreshold(float t) { conf_threshold_ = t; }
    void setNmsThreshold(float t) { nms_threshold_ = t; }

    std::vector<ArmorDetection> detect(const cv::Mat& frame);

private:
    // letterbox 缩放：保持长宽比塞进网络输入，空位填灰边，不能用 resize 直接拉伸
    struct Preprocess {
        cv::Mat blob;      // NCHW float 数据
        float scale = 1.f;
        int pad_x = 0;     // 左右各补的灰边宽度
        int pad_y = 0;     // 上下各补的灰边高度
    };
    Preprocess letterbox(const cv::Mat& frame) const;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;

    // ORT 的 Run() 要 char* 数组，所以名字存两份：一份持有内存，一份交给 ORT
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;

    std::vector<int64_t> input_shape_; // 从模型里读，不写死
    cv::Size input_size_;              // 网络输入尺寸（宽, 高）

    float conf_threshold_ = kDefaultConfidence;
    float nms_threshold_ = kDefaultNms;
    bool loaded_ = false;
};
