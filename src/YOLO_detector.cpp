#include "YOLO_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

static constexpr int kNumClasses = 9;

// 检测网络按兵种分的 9 类，与数字网络的 9 类不是同一套，不能互相翻译
static const char* kArmorClassNames[kNumClasses] = {
    "sentry",     // 0 哨兵
    "hero",       // 1 英雄
    "engineer",   // 2 工程
    "infantry3",  // 3 三号步兵
    "infantry4",  // 4 四号步兵
    "infantry5",  // 5 五号步兵（v8n-20260726 已不再识别）
    "outpost",    // 6 前哨站
    "base",       // 7 基地（底部）
    "base_top",   // 8 基地（顶部）
};

// 只有 3/4/5 号步兵的装甲板上印的是数字，其余兵种印图标，返回 -1
static int digitOfClass(int class_id) {
    return (class_id >= 3 && class_id <= 5) ? class_id : -1;
}

YOLODetector::~YOLODetector() = default;

bool YOLODetector::load(const std::string& onnx_path) {
    loaded_ = false;
    session_.reset();
    env_.reset();

    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rm_vision_hw");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1); // 单线程，避免逐帧推理时和主线程抢核
        // EXTENDED 里包含"给 fp16 节点自动插 Cast 转 fp32"，这个模型是 fp16 权重，
        // CPU 后端靠这一步才能跑起来，不能改成 DISABLE_ALL。
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), opts);
    } catch (const Ort::Exception& e) {
        std::cerr << "[警告] 加载装甲板检测模型失败: " << onnx_path << "\n"
                  << "       ONNX Runtime 报错: " << e.what() << "\n"
                  << "       常见原因：1) 文件是 0 字节占位；2) 模型 IR/opset 版本高于本机 ORT 支持范围\n"
                  << "       （详见 models/README.md 里「armor.onnx 改过版本号」一节）。\n";
        return false;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    const size_t num_inputs = session_->GetInputCount();
    if (num_inputs == 0) {
        std::cerr << "[警告] 模型没有任何输入节点: " << onnx_path << "\n";
        return false;
    }
    input_names_.clear();
    for (size_t i = 0; i < num_inputs; ++i) {
        // 返回的是带自定义删除器的指针，必须立刻拷成 std::string，否则出作用域就失效
        Ort::AllocatedStringPtr name = session_->GetInputNameAllocated(i, allocator);
        input_names_.emplace_back(name.get());
    }
    const Ort::TypeInfo input_info = session_->GetInputTypeInfo(0);
    input_shape_ = input_info.GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape_.size() != 4) {
        std::cerr << "[警告] 期望模型输入是 4 维 NCHW，实际是 "
                  << input_shape_.size() << " 维\n";
        return false;
    }
    input_size_ = cv::Size(static_cast<int>(input_shape_[3]), static_cast<int>(input_shape_[2]));

    const size_t num_outputs = session_->GetOutputCount();
    output_names_.clear();
    for (size_t i = 0; i < num_outputs; ++i) {
        Ort::AllocatedStringPtr name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(name.get());
    }
    if (num_outputs == 0) {
        std::cerr << "[警告] 模型没有任何输出节点: " << onnx_path << "\n";
        return false;
    }

    // 指针数组只在名字容器填完之后建，中途不能再改动容器
    input_name_ptrs_.clear();
    for (const auto& n : input_names_) input_name_ptrs_.push_back(n.c_str());
    output_name_ptrs_.clear();
    for (const auto& n : output_names_) output_name_ptrs_.push_back(n.c_str());

    loaded_ = true;
    std::cout << "[信息] 装甲板检测模型已加载: " << onnx_path << "\n"
              << "       输入: " << input_shape_[0] << "×" << input_shape_[1]
              << "×" << input_shape_[2] << "×" << input_shape_[3] << " (NCHW)\n"
              << "       输出: " << num_outputs << " 个节点，取第 1 个「"
              << output_names_[0] << "」\n";
    return true;
}

YOLODetector::Preprocess YOLODetector::letterbox(const cv::Mat& frame) const {
    Preprocess out;

    const int src_w = frame.cols;
    const int src_h = frame.rows;
    const int dst_w = input_size_.width;
    const int dst_h = input_size_.height;

    // 取较小的缩放比，保证整幅图都塞得进去
    out.scale = std::min(static_cast<float>(dst_w) / src_w,
                         static_cast<float>(dst_h) / src_h);
    const int new_w = cvRound(src_w * out.scale);
    const int new_h = cvRound(src_h * out.scale);
    // 补边均分两侧，逆变换用的是同一个 pad 值，不会引入偏差
    out.pad_x = (dst_w - new_w) / 2;
    out.pad_y = (dst_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 灰边填 124，是该模型部署时的填充值（不是 Ultralytics 默认的 114），填错会让边界附近响应偏移
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, out.pad_y, dst_h - new_h - out.pad_y,
                       out.pad_x, dst_w - new_w - out.pad_x,
                       cv::BORDER_CONSTANT, cv::Scalar(124, 124, 124));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    cv::Mat rgb_f;
    rgb.convertTo(rgb_f, CV_32F, 1.0 / 255.0);

    // HWC → CHW：先 split 出三个单通道，再拷进 NCHW 的大块内存
    std::vector<cv::Mat> channels;
    cv::split(rgb_f, channels);

    const int sz[4] = {1, 3, dst_h, dst_w};
    out.blob = cv::Mat(4, sz, CV_32F);
    const size_t plane_bytes = static_cast<size_t>(dst_h) * dst_w * sizeof(float);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(out.blob.ptr<float>(0, c), channels[c].ptr<float>(0), plane_bytes);
    }

    return out;
}

std::vector<ArmorDetection> YOLODetector::detect(const cv::Mat& frame) {
    std::vector<ArmorDetection> results;
    if (!loaded_ || frame.empty()) return results;

    // 1. 预处理
    const Preprocess pre = letterbox(frame);
    const size_t total_floats = pre.blob.total();

    // 2. 推理
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, reinterpret_cast<float*>(pre.blob.data), total_floats,
        input_shape_.data(), input_shape_.size());

    std::vector<Ort::Value> outputs;
    try {
        outputs = session_->Run(Ort::RunOptions{nullptr},
                                input_name_ptrs_.data(), &input_tensor, 1,
                                output_name_ptrs_.data(),
                                static_cast<size_t>(output_name_ptrs_.size()));
    } catch (const Ort::Exception& e) {
        std::cerr << "[错误] 神经网络推理失败: " << e.what() << "\n";
        return results;
    }
    if (outputs.empty() || !outputs[0].IsTensor()) return results;

    // 3. 解析输出
    const auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> out_shape = out_info.GetShape();

    // 输出不一定是 fp32。若留成 fp16，用 GetTensorMutableData<float>() 读会把两个 fp16
    // 当成一个 float32 解释，数值全乱且不报错，所以先看元素类型再决定怎么取。
    std::vector<float> converted;
    const float* data = nullptr;
    const auto elem_type = out_info.GetElementType();
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        data = outputs[0].GetTensorMutableData<float>();
    } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const Ort::Float16_t* half = outputs[0].GetTensorMutableData<Ort::Float16_t>();
        const size_t n = out_info.GetElementCount();
        converted.resize(n);
        for (size_t i = 0; i < n; ++i) converted[i] = half[i].ToFloat();
        data = converted.data();
    } else {
        std::cerr << "[错误] 输出张量元素类型既不 float32 也不 float16（枚举值 "
                  << static_cast<int>(elem_type) << "），无法解析\n";
        return results;
    }
    if (out_shape.size() != 3) {
        std::cerr << "[错误] 期望输出是 3 维，实际是 " << out_shape.size() << " 维，无法解析\n";
        return results;
    }

    // 两种排布都支持：看第 1、2 维哪个是通道数（21 或 22）这个小值
    const int64_t d1 = out_shape[1];
    const int64_t d2 = out_shape[2];
    int num_anchors = 0, channels = 0; bool row_major = false;
    if (d2 == 21 || d2 == 22) {
        num_anchors = static_cast<int>(d1);
        channels = static_cast<int>(d2);
        row_major = true;
    } else if (d1 == 21 || d1 == 22) {
        num_anchors = static_cast<int>(d2);
        channels = static_cast<int>(d1);
        row_major = false;
    } else {
        std::cerr << "[错误] 无法识别的输出通道数: " << d1 << "×" << d2
                  << "（期望出现 21 或 22，请确认模型是装甲板关键点检测模型）\n";
        return results;
    }

    // 两种布局的通道偏移见头文件。它们都没有 bbox 回归，框是 4 个关键点的外接矩形，
    // 只用于 NMS；按 [框][关键点][类别] 去解析会把坐标当成分数，候选全部过阈值。
    const bool is_v8 = (channels == 21);
    const int kpt_off = is_v8 ? 13 : 0;
    const int cls_off = is_v8 ? 4 : 13;
    const int obj_off = is_v8 ? -1 : 8;

    auto at = [&](int anchor, int c) -> float {
        return row_major ? data[static_cast<size_t>(anchor) * channels + c]
                         : data[static_cast<size_t>(c) * num_anchors + anchor];
    };

    // 4. 置信度过滤
    struct Candidate {
        float kpt[8] = {0};  // 4 个关键点 (x,y)，输入画布坐标
        float conf = 0.f;
        int class_id = 0;
        float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f; // 关键点外接矩形，仅 NMS 用
    };
    std::vector<Candidate> cands;
    cands.reserve(64);

    for (int a = 0; a < num_anchors; ++a) {
        // 类别分取 9 类里最大的那个
        int best_cls = 0;
        float best_score = at(a, cls_off);
        for (int c = 1; c < kNumClasses; ++c) {
            const float s = at(a, cls_off + c);
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        if (!(best_score > 0.f)) continue; // 顺带挡掉 NaN

        float conf = best_score;
        if (!is_v8) {
            // V5 的 obj 是 logit，先 sigmoid 再乘类别分
            const float logit = at(a, obj_off);
            conf *= 1.f / (1.f + std::exp(-logit));
        }
        if (conf < conf_threshold_) continue;

        Candidate cd;
        cd.conf = conf;
        cd.class_id = best_cls;
        for (int k = 0; k < 8; ++k) cd.kpt[k] = at(a, kpt_off + k);

        // 由 4 个关键点算外接矩形，长宽太小说明四点几乎重合，是退化候选
        cd.x1 = std::min(std::min(cd.kpt[0], cd.kpt[2]), std::min(cd.kpt[4], cd.kpt[6]));
        cd.x2 = std::max(std::max(cd.kpt[0], cd.kpt[2]), std::max(cd.kpt[4], cd.kpt[6]));
        cd.y1 = std::min(std::min(cd.kpt[1], cd.kpt[3]), std::min(cd.kpt[5], cd.kpt[7]));
        cd.y2 = std::max(std::max(cd.kpt[1], cd.kpt[3]), std::max(cd.kpt[5], cd.kpt[7]));
        if (cd.x2 - cd.x1 < 1.f || cd.y2 - cd.y1 < 1.f) continue;

        cands.push_back(cd);
    }

    if (cands.empty()) return results;

    // 5. NMS 去重，同一块板常被相邻几个锚点同时命中
    auto iou = [](const Candidate& a, const Candidate& b) {
        const float x1 = std::max(a.x1, b.x1);
        const float y1 = std::max(a.y1, b.y1);
        const float x2 = std::min(a.x2, b.x2);
        const float y2 = std::min(a.y2, b.y2);
        const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
        const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        const float uni = area_a + area_b - inter;
        return (uni > 0.f) ? (inter / uni) : 0.f;
    };

    std::vector<int> order(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return cands[a].conf > cands[b].conf;
    });

    std::vector<bool> suppressed(cands.size(), false);
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const int i = order[oi];
        if (suppressed[i]) continue;
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            const int j = order[oj];
            if (suppressed[j]) continue;
            if (iou(cands[i], cands[j]) > nms_threshold_) suppressed[j] = true;
        }
    }

    // 6. 逆变换回原图坐标。letterbox 有缩放和补边两步，
    //    漏 pad 会让框整体偏移几十像素，漏 scale 会让框尺寸全错。
    auto to_original = [&](float x, float y) -> cv::Point2f {
        return cv::Point2f((x - static_cast<float>(pre.pad_x)) / pre.scale,
                           (y - static_cast<float>(pre.pad_y)) / pre.scale);
    };

    for (size_t i = 0; i < cands.size(); ++i) {
        if (suppressed[i]) continue;

        ArmorDetection det;
        det.method = "nn";
        det.confidence = cands[i].conf;
        det.class_id = cands[i].class_id;
        det.class_name = kArmorClassNames[cands[i].class_id];
        det.digit = digitOfClass(cands[i].class_id);

        // 关键点重排：网络是 0左上 1左下 2右下 3右上，本项目是 0左上 1右上 2右下 3左下
        det.image_points[0] = to_original(cands[i].kpt[0], cands[i].kpt[1]); // 左上
        det.image_points[1] = to_original(cands[i].kpt[6], cands[i].kpt[7]); // 右上
        det.image_points[2] = to_original(cands[i].kpt[4], cands[i].kpt[5]); // 右下
        det.image_points[3] = to_original(cands[i].kpt[2], cands[i].kpt[3]); // 左下

        // 关键点跑出画面的基本可以确定是误检
        bool inside = true;
        for (const auto& p : det.image_points) {
            if (p.x < 0.f || p.y < 0.f || p.x >= frame.cols || p.y >= frame.rows) {
                inside = false;
                break;
            }
        }
        if (!inside) continue;

        results.push_back(det);
    }

    return results;
}
