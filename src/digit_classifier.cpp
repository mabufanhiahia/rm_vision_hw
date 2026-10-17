#include "digit_classifier.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {

// 拉正后的目标尺寸，照抄 rm_auto_aim 的常量，改了识别率会明显下降
constexpr int kWarpHeight = 28;
constexpr int kSmallArmorWidth = 32;
constexpr int kLargeArmorWidth = 54;
constexpr int kLightLength = 12;

// 只能写 const 不能写 constexpr：cv::Size 的构造函数不是 constexpr 的，写 constexpr 编译不过
const cv::Size kRoiSize(20, 28);

} // namespace

bool DigitClassifier::load(const std::string& onnx_path, const std::string& label_path) {
    labels_.clear();
    negative_id_ = -1;
    loaded_ = false;

    std::ifstream fin(label_path);
    if (!fin.is_open()) {
        std::cerr << "[警告] 读不到标签文件: " << label_path << "\n";
        return false;
    }
    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // 兼容 CRLF
        if (line.empty()) continue;
        labels_.push_back(line);
    }
    if (labels_.empty()) {
        std::cerr << "[警告] 标签文件是空的: " << label_path << "\n";
        return false;
    }
    for (size_t i = 0; i < labels_.size(); ++i) { // 自动定位 negative，不写死下标 8
        if (labels_[i] == "negative" || labels_[i] == "-1") negative_id_ = static_cast<int>(i);
    }

    try {
        net_ = cv::dnn::readNetFromONNX(onnx_path);
    } catch (const cv::Exception& e) {
        std::cerr << "[警告] 加载数字识别模型失败: " << onnx_path << "\n"
                  << "       OpenCV 报错: " << e.what() << "\n"
                  << "       常见原因：文件是 0 字节的占位文件（还没真的下载），或者格式不对。\n";
        return false;
    }
    if (net_.empty()) {
        std::cerr << "[警告] 数字识别模型加载后为空: " << onnx_path << "\n"
                  << "       请确认文件已经真的下载下来（不是 0 字节）。\n";
        return false;
    }

    loaded_ = true;
    std::cout << "[信息] 数字识别模型已加载: " << onnx_path
              << "（" << labels_.size() << " 类: ";
    for (size_t i = 0; i < labels_.size(); ++i) {
        std::cout << (i ? "," : "") << labels_[i];
    }
    std::cout << "）\n";
    return true;
}

// 把给定的四个点拉正成 20×28 黑白数字图，src 顺序为 左下 → 左上 → 右上 → 右下
cv::Mat DigitClassifier::warpNumberROI(const cv::Mat& frame,
                                       const cv::Point2f src[4]) const {
    if (frame.empty()) return cv::Mat();

    // 小板 135×55、大板 225×55，图像上"上边长/灯条长"的比值差别明显
    const double top_len = cv::norm(src[1] - src[0]);
    const double left_len = cv::norm(src[3] - src[0]);
    if (left_len <= 1.0) return cv::Mat(); // 框退化成一个点，没法变换
    const double ratio = top_len / left_len;
    const int warp_width = (ratio > 2.7) ? kLargeArmorWidth : kSmallArmorWidth;

    // 目标四边形：灯条竖直摆放、上下居中，数字区域自然落在正中
    const int top_y = (kWarpHeight - kLightLength) / 2 - 1;
    const int bottom_y = top_y + kLightLength;
    cv::Point2f dst[4] = {
        cv::Point2f(0.f, static_cast<float>(bottom_y)),                 // 左下
        cv::Point2f(0.f, static_cast<float>(top_y)),                    // 左上
        cv::Point2f(static_cast<float>(warp_width - 1), static_cast<float>(top_y)),    // 右上
        cv::Point2f(static_cast<float>(warp_width - 1), static_cast<float>(bottom_y))  // 右下
    };

    cv::Mat warped;
    const cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::warpPerspective(frame, warped, M, cv::Size(warp_width, kWarpHeight));

    // 取中间那块，数字就印在两灯条之间
    cv::Mat roi = warped(cv::Rect(cv::Point((warp_width - kRoiSize.width) / 2, 0), kRoiSize));

    // 不能 cvtColor(roi, roi, ...)：原地改通道数不安全，所以单独开一个 gray
    cv::Mat gray;
    if (roi.channels() == 3) {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = roi;
    }

    // OTSU 自适应二值化：灯条亮度随距离与曝光变化很大，固定阈值顾此失彼
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return gray;
}

cv::Mat DigitClassifier::extractNumberImage(const cv::Mat& frame,
                                            const ArmorDetection& armor) const {
    if (frame.empty()) return cv::Mat();

    // image_points 是 左上→右上→右下→左下，这里按 左下→左上→右上→右下 重排
    const auto& p = armor.image_points;
    const cv::Point2f src[4] = {
        p[3], // 左下
        p[0], // 左上
        p[1], // 右上
        p[2]  // 右下
    };
    return warpNumberROI(frame, src);
}

// 扶正版：两灯条中心连线为横轴、灯条长度均值为高重建四角，得到矩形，抹掉梯形造成的倾斜。
// 横轴自带板子的滚转角，所以不是把板强行掰平，只是消掉左右灯条长短不一的影响。
cv::Mat DigitClassifier::extractNumberImageStraight(const cv::Mat& frame,
                                                    const ArmorDetection& armor) const {
    if (frame.empty()) return cv::Mat();
    const auto& p = armor.image_points;

    const cv::Point2f lc = (p[0] + p[3]) * 0.5f; // 左灯条中心
    const cv::Point2f rc = (p[1] + p[2]) * 0.5f; // 右灯条中心

    cv::Point2f x_axis = rc - lc;
    const float n = static_cast<float>(cv::norm(x_axis));
    if (n < 1e-3f) return cv::Mat();
    x_axis /= n;

    // 横轴转 90° 得纵轴；图像 y 向下，故 (-x.y, x.x) 指向下方
    const cv::Point2f y_axis(-x_axis.y, x_axis.x);

    // 高度取两条边的平均值，两次测量互相纠偏
    const float h = 0.5f * (static_cast<float>(cv::norm(p[3] - p[0])) +
                            static_cast<float>(cv::norm(p[2] - p[1])));
    const cv::Point2f half_y = y_axis * (h * 0.5f);

    const cv::Point2f src[4] = {
        lc + half_y,  // 左下
        lc - half_y,  // 左上
        rc - half_y,  // 右上
        rc + half_y   // 右下
    };
    return warpNumberROI(frame, src);
}

bool DigitClassifier::runInference(const cv::Mat& number, int& class_id, float& prob) const {
    // 网络训练时喂的是 0/1 二值图，这一步不能省
    cv::Mat normalized;
    number.convertTo(normalized, CV_32F, 1.0 / 255.0);

    cv::Mat blob;
    cv::dnn::blobFromImage(normalized, blob);
    net_.setInput(blob);
    const cv::Mat output = net_.forward(); // 1×9 未归一化 logits

    // softmax 先减最大值再取 exp，避免 exp(大数) 溢出；结果与平移无关，是等价的
    const cv::Mat out_1d = output.reshape(1, 1);
    double max_logit = 0.0;
    cv::minMaxLoc(out_1d, nullptr, &max_logit);
    cv::Mat exp_out;
    cv::exp(out_1d - max_logit, exp_out);
    const cv::Mat prob_mat = exp_out / cv::sum(exp_out)[0];

    cv::Point max_loc;
    double max_prob = 0.0;
    cv::minMaxLoc(prob_mat, nullptr, &max_prob, nullptr, &max_loc);
    class_id = max_loc.x;
    prob = static_cast<float>(max_prob);
    return true;
}

int DigitClassifier::classifyKeepNegative(const cv::Mat& frame,
                                          const ArmorDetection& armor,
                                          float* confidence) const {
    if (confidence != nullptr) *confidence = 0.0f;
    if (!loaded_) return -1;

    cv::Mat number = extractNumberImage(frame, armor);
    if (number.empty()) return -1;

    int class_id = -1;
    float prob = 0.f;
    if (!runInference(number, class_id, prob)) return -1;

    // 有把握地判 negative 时换扶正版 ROI 再问一次——丢弃框的代价比多跑一次推理（<1ms）大得多。
    // 只有扶正后确实认出了非 negative 类别才改判，否则保留 negative，不然跨板假框又会被放回来。
    if (isNegative(class_id) && prob >= threshold_) {
        const cv::Mat straight = extractNumberImageStraight(frame, armor);
        if (!straight.empty()) {
            int id2 = -1;
            float prob2 = 0.f;
            if (runInference(straight, id2, prob2) && !isNegative(id2) && prob2 >= threshold_) {
                if (confidence != nullptr) *confidence = prob2;
                return id2;
            }
        }
    }

    if (confidence != nullptr) *confidence = prob;
    return class_id; // negative 也照实返回，交给调用方处置
}

int DigitClassifier::classify(const cv::Mat& frame, const ArmorDetection& armor,
                              float* confidence) const {
    float prob = 0.f;
    const int class_id = classifyKeepNegative(frame, armor, &prob);
    if (confidence != nullptr) *confidence = prob;

    if (class_id < 0) return -1;         // 推理失败
    if (prob < threshold_) return -1;    // 没把握，当作没认出来
    if (isNegative(class_id)) return -1; // 明确说这里没有数字
    return class_id;
}

std::string DigitClassifier::labelName(int class_id) const {
    if (class_id < 0 || class_id >= static_cast<int>(labels_.size())) return std::string();
    return labels_[class_id];
}
