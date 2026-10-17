#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 数字识别：给一块框好的装甲板，用开源 MLP（models/digit.onnx）识别上面的数字或兵种。
// 标签 9 类：0~4 是数字 1~5，5 outpost / 6 guard / 7 base，8 negative 表示区域内没有数字。
class DigitClassifier {
public:
    static constexpr float kDefaultThreshold = 0.70f; // softmax 最大概率低于它就算没把握

    // 加载模型与标签。失败返回 false，传统方案会退化成不识别数字，仍能检测和解算。
    bool load(const std::string& onnx_path, const std::string& label_path);

    bool loaded() const { return loaded_; }
    const std::vector<std::string>& labels() const { return labels_; }

    void setThreshold(float t) { threshold_ = t; }
    float threshold() const { return threshold_; }

    // 返回类别下标 0~8（不是数字本身）；判为 negative、置信度不足、推理失败都返回 -1
    int classify(const cv::Mat& frame, const ArmorDetection& armor,
                 float* confidence = nullptr) const;

    // 与 classify() 的区别是保留 negative 的返回，让调用方自己区分"确实没有"和"没把握"
    int classifyKeepNegative(const cv::Mat& frame, const ArmorDetection& armor,
                             float* confidence = nullptr) const;

    // 类别下标转数字：只有 0~4 对应数字 1~5，其余返回 -1
    static int digitOf(int class_id) { return (class_id >= 0 && class_id <= 4) ? class_id + 1 : -1; }

    std::string labelName(int class_id) const;

    bool isNegative(int class_id) const { return class_id >= 0 && class_id == negative_id_; }

    // 单独暴露预处理，调不出数字时可以把这张 20×28 的图存出来看
    cv::Mat extractNumberImage(const cv::Mat& frame, const ArmorDetection& armor) const;

    // 扶正版：以两灯条中心连线为横轴重建矩形再拉正，抹掉左右灯条长短不一造成的梯形
    cv::Mat extractNumberImageStraight(const cv::Mat& frame,
                                       const ArmorDetection& armor) const;

private:
    // 把四个点（左下→左上→右上→右下）拉正成 20×28 黑白数字图，两种取点方式共用
    cv::Mat warpNumberROI(const cv::Mat& frame, const cv::Point2f src[4]) const;

    bool runInference(const cv::Mat& number, int& class_id, float& prob) const;

    mutable cv::dnn::Net net_; // classify() 是 const，但 setInput/forward 非 const，故加 mutable
    std::vector<std::string> labels_;
    int negative_id_ = -1;     // negative 的下标，从 label.txt 自动找
    float threshold_ = kDefaultThreshold;
    bool loaded_ = false;
};
