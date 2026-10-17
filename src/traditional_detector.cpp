#include "traditional_detector.h"

#include <algorithm>
#include <cmath>

namespace {

// 蓝方掩膜：B-G >= min_dom_g 且 B-R >= min_dom_r 且 V >= min_v。
// subtract 对 CV_8U 饱和，负值截成 0；threshold 是严格大于，故阈值取下限-1 保持闭区间。
cv::Mat buildBlueMask(const cv::Mat& frame, double min_dom_g, double min_dom_r, double min_v) {
    std::vector<cv::Mat> bgr;
    cv::split(frame, bgr);
    const cv::Mat& B = bgr[0];
    const cv::Mat& G = bgr[1];
    const cv::Mat& R = bgr[2];

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_ch;
    cv::split(hsv, hsv_ch);

    cv::Mat diff_bg, diff_br, m1, m2, m3, mask;
    cv::subtract(B, G, diff_bg);
    cv::subtract(B, R, diff_br);

    cv::threshold(diff_bg, m1, min_dom_g - 1.0, 255.0, cv::THRESH_BINARY);
    cv::threshold(diff_br, m2, min_dom_r - 1.0, 255.0, cv::THRESH_BINARY);
    cv::threshold(hsv_ch[2], m3, min_v - 1.0, 255.0, cv::THRESH_BINARY);

    cv::bitwise_and(m1, m2, mask);
    cv::bitwise_and(mask, m3, mask);
    return mask;
}

// 红方掩膜：R-G >= min_dom_g 且 R-B >= min_dom_b 且 V >= min_v。视频是红方，只留蓝色会一帧都检不出。
cv::Mat buildRedMask(const cv::Mat& frame, double min_dom_g, double min_dom_b, double min_v) {
    std::vector<cv::Mat> bgr;
    cv::split(frame, bgr);
    const cv::Mat& B = bgr[0];
    const cv::Mat& G = bgr[1];
    const cv::Mat& R = bgr[2];

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_ch;
    cv::split(hsv, hsv_ch);

    cv::Mat diff_rg, diff_rb, m1, m2, m3, mask;
    cv::subtract(R, G, diff_rg);
    cv::subtract(R, B, diff_rb);

    cv::threshold(diff_rg, m1, min_dom_g - 1.0, 255.0, cv::THRESH_BINARY);
    cv::threshold(diff_rb, m2, min_dom_b - 1.0, 255.0, cv::THRESH_BINARY);
    cv::threshold(hsv_ch[2], m3, min_v - 1.0, 255.0, cv::THRESH_BINARY);

    cv::bitwise_and(m1, m2, mask);
    cv::bitwise_and(mask, m3, mask);
    return mask;
}

// 过曝补光掩膜。灯条被打到全白时颜色判据在整根上都失效，但过曝是逐通道依次饱和的：
// 红方 R 先满、然后 G、最后 B，所以过曝区总是"主通道饱和 + 另一通道滞后"，白色物体 R-B≈0 过不了。
// 没有这一刀时同一块板会一会儿只剩一根灯条，框就跟着一闪一闪。
cv::Mat buildOverexposedMask(const cv::Mat& frame, double min_sat, double min_lag, double min_mid) {
    std::vector<cv::Mat> bgr;
    cv::split(frame, bgr);
    const cv::Mat& B = bgr[0];
    const cv::Mat& G = bgr[1];
    const cv::Mat& R = bgr[2];

    cv::Mat t, m_sat, m_lag, m_mid, part, mask;

    // 过曝红
    cv::threshold(R, m_sat, min_sat - 1.0, 255.0, cv::THRESH_BINARY);
    cv::subtract(R, B, t);
    cv::threshold(t, m_lag, min_lag - 1.0, 255.0, cv::THRESH_BINARY);
    cv::threshold(G, m_mid, min_mid - 1.0, 255.0, cv::THRESH_BINARY);
    cv::bitwise_and(m_sat, m_lag, part);
    cv::bitwise_and(part, m_mid, mask);

    // 过曝蓝
    cv::threshold(B, m_sat, min_sat - 1.0, 255.0, cv::THRESH_BINARY);
    cv::subtract(B, R, t);
    cv::threshold(t, m_lag, min_lag - 1.0, 255.0, cv::THRESH_BINARY);
    cv::threshold(G, m_mid, min_mid - 1.0, 255.0, cv::THRESH_BINARY);
    cv::bitwise_and(m_sat, m_lag, part);
    cv::bitwise_and(part, m_mid, part);
    cv::bitwise_or(mask, part, mask);

    return mask;
}

} // namespace

TraditionalDetector::TraditionalDetector() {}

// 倾角用最长边的方向算。不能直接用 rect.angle（定义随版本变化，竖直矩形给 90），
// 也不能用 top-bottom 连线（水平细长条会算成 0°，看着像竖直灯条骗过筛选）。
TraditionalDetector::LightBar TraditionalDetector::makeLightBar(const cv::RotatedRect& rect) {
    LightBar bar;
    bar.rect = rect;
    cv::Point2f p[4];
    rect.points(p);

    // 取最长的那条边定方向：竖直 → 0°，水平 → 90°
    double best_len = -1.0;
    double tilt = 90.0;
    for (int i = 0; i < 4; ++i) {
        const cv::Point2f d = p[(i + 1) % 4] - p[i];
        const double len = std::sqrt(static_cast<double>(d.x) * d.x +
                                     static_cast<double>(d.y) * d.y);
        if (len > best_len) {
            best_len = len;
            tilt = std::atan2(std::abs(static_cast<double>(d.x)),
                              std::abs(static_cast<double>(d.y))) *
                   180.0 / CV_PI;
        }
    }
    bar.tilt = static_cast<float>(tilt);

    // 确认竖直之后，按 y 排序取上下端点
    std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    bar.top = (p[0] + p[1]) * 0.5f;
    bar.bottom = (p[2] + p[3]) * 0.5f;
    bar.length = static_cast<float>(cv::norm(bar.top - bar.bottom));
    bar.width = static_cast<float>(cv::norm(p[0] - p[1]));
    return bar;
}

// 面积、长宽比、倾角三道筛
bool TraditionalDetector::isLightBar(const LightBar& bar) const {
    const double area = static_cast<double>(bar.rect.size.width) * bar.rect.size.height;
    if (area < min_area_ || area > max_area_) return false;

    const double w = bar.rect.size.width;
    const double h = bar.rect.size.height;
    const double aspect_ratio = std::max(w, h) / std::max(std::min(w, h), 1e-6);
    if (aspect_ratio < min_aspect_ratio_ || aspect_ratio > max_aspect_ratio_) return false;

    return bar.tilt <= 45.0f;
}

// 打分：主项是 |中心距/灯条长 − 2.45|，次项是长度不一致与中心连线倾斜；返回负数表示不可能配对。
double TraditionalDetector::pairCost(const LightBar& a, const LightBar& b) const {
    const double ha = a.length;
    const double hb = b.length;
    const double hmin = std::min(ha, hb);
    const double hmax = std::max(ha, hb);
    if (hmin <= 0.0) return -1.0;

    if (hmax / hmin > 2.2) return -1.0; // 远处灯条只有几像素，长度噪声大，故放宽

    // 用上端点连线，两灯条等长时与中心连线等价
    const double dx = static_cast<double>(b.top.x) - a.top.x;
    const double dy = static_cast<double>(b.top.y) - a.top.y;
    const double raw_angle = std::abs(std::atan2(dy, dx)) * 180.0 / CV_PI;
    // atan2 落在 [0,180]，折到 [0,90]，否则左右灯条顺序一反就误判
    const double folded = (raw_angle > 90.0) ? (180.0 - raw_angle) : raw_angle;
    if (folded > kMaxCenterLineAngle) return -1.0;

    // 分母取两根灯条的平均长度，不是较短的那根：image1 真同板的两根是 13px 与 8px，
    // 用 8px 做分母比值虚高到 3.32，取平均 10.5px 是 2.53，正好落在先验 2.45 附近。
    const double dist = cv::norm(a.rect.center - b.rect.center);
    const double hlen = 0.5 * (ha + hb);
    const double ratio = dist / hlen;
    if (ratio < 1.0 || ratio > kMaxBarDistRatio) return -1.0;

    double cost = std::abs(ratio - kLightBarDistRatio);

    // 长度差按绝对像素容差罚，不按比例：远处灯条只有几像素，误差是绝对量，
    // 按比例罚会把真配对压下去，让长度恰好相同的跨板组合胜出。
    const double len_excess = std::max(0.0, std::abs(ha - hb) - 3.0);
    cost += 0.3 * len_excess / hlen;

    cost += 0.5 * folded / 90.0;
    return cost;
}

// 同板两根灯条之间不该夹着第三根，夹着说明这两根分属相邻的两块板
bool TraditionalDetector::containLight(const LightBar& a, const LightBar& b,
                                       const std::vector<LightBar>& all) const {
    const std::vector<cv::Point2f> pts = {a.top, a.bottom, b.top, b.bottom};
    const cv::Rect box = cv::boundingRect(pts);

    for (const auto& other : all) {
        const bool is_a = cv::norm(other.rect.center - a.rect.center) < 1e-3;
        const bool is_b = cv::norm(other.rect.center - b.rect.center) < 1e-3;
        if (is_a || is_b) continue;

        const cv::Point top_i(cvRound(other.top.x), cvRound(other.top.y));
        const cv::Point bottom_i(cvRound(other.bottom.x), cvRound(other.bottom.y));
        const cv::Point center_i(cvRound(other.rect.center.x), cvRound(other.rect.center.y));
        if (box.contains(top_i) || box.contains(bottom_i) || box.contains(center_i)) {
            return true;
        }
    }
    return false;
}

std::vector<ArmorDetection> TraditionalDetector::detect(const cv::Mat& frame,
                                                        cv::Mat* debug_mask) {
    std::vector<ArmorDetection> results;
    if (frame.empty()) return results;

    // 1. 颜色分割。蓝红两套掩膜取或，同一帧一般只有一方，互不干扰。
    const cv::Mat mask_blue =
        buildBlueMask(frame, min_blue_dom_g_, min_blue_dom_r_, min_v_);
    const cv::Mat mask_red =
        buildRedMask(frame, min_red_dom_g_, min_red_dom_b_, min_v_);
    cv::Mat mask;
    cv::bitwise_or(mask_blue, mask_red, mask);

    // 过曝补光：把被打白、主判据失效的灯条捞回来
    const cv::Mat mask_ov =
        buildOverexposedMask(frame, min_ov_sat_, min_ov_lag_, min_ov_mid_);
    cv::bitwise_or(mask, mask_ov, mask);

    if (debug_mask != nullptr) *debug_mask = mask.clone();

    // 2. 形态学。核取竖向 3×5：灯条竖直，能把被过曝切成虚线的部分接成整根，
    //    又不会横向把相邻两根粘在一起。闭运算不改变实心块尺寸，对图片素材无影响。
    const cv::Mat k_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k_close, cv::Point(-1, -1), 2);

    // 填实空心轮廓，否则面积筛选和调试图都会失真
    {
        std::vector<std::vector<cv::Point>> holes;
        cv::findContours(mask.clone(), holes, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(mask, holes, -1, cv::Scalar(255), cv::FILLED);
    }

    const cv::Mat k_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k_open);

    // 3. 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 4. 筛灯条
    std::vector<LightBar> light_bars;
    for (const auto& cnt : contours) {
        if (cnt.size() < 4) continue;
        const LightBar bar = makeLightBar(cv::minAreaRect(cnt));
        if (isLightBar(bar)) light_bars.push_back(bar);
    }

    // 5. 配对：所有可行组合打分，再从优到劣依次占用，保证一根灯条只属于一块板
    struct Pair {
        int i, j;
        double cost;
    };
    const int n = static_cast<int>(light_bars.size());
    std::vector<Pair> pairs;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double c = pairCost(light_bars[i], light_bars[j]);
            if (c < 0.0) continue;
            if (containLight(light_bars[i], light_bars[j], light_bars)) continue;
            pairs.push_back(Pair{i, j, c});
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.cost < b.cost; });

    std::vector<bool> used(n, false);
    for (const Pair& p : pairs) {
        if (used[p.i] || used[p.j]) continue;
        used[p.i] = used[p.j] = true;

        // 统一左灯条在前、右灯条在后
        const LightBar& A = light_bars[p.i];
        const LightBar& B = light_bars[p.j];
        const LightBar& left = (A.rect.center.x < B.rect.center.x) ? A : B;
        const LightBar& right = (A.rect.center.x < B.rect.center.x) ? B : A;

        ArmorDetection armor;
        armor.method = "traditional";
        armor.digit = -1; // 数字由检测器集合接数字网络补上

        // 四角取灯条真实上下端点，保留灯条自身倾斜。比外推端正矩形更准：
        // image1 上端正矩形重投影 0.016px 但三维误差 371mm，端点法重投影 1.243px 误差 269mm，
        // 因为人为构造的完美矩形 PnP 必然拟合得好，误差被藏起来了。
        armor.image_points = {
            left.top,     // 左上
            right.top,    // 右上
            right.bottom, // 右下
            left.bottom   // 左下
        };

        results.push_back(armor);
    }

    return results;
}
