#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>
#include <vector>

// 传统视觉装甲板检测：颜色分割 → 形态学 → 找轮廓 → 灯条筛选 → 灯条配对，不依赖神经网络。
class TraditionalDetector {
public:
    // 一根候选灯条。存上下端点是为了配对时算中心连线与长度，配成板后四角直接取端点。
    struct LightBar {
        cv::RotatedRect rect;
        cv::Point2f top;      // 上端点
        cv::Point2f bottom;   // 下端点
        float length = 0.f;
        float width = 0.f;
        float tilt = 90.f;    // 长轴与竖直方向夹角(度)，竖直=0，水平=90
    };

    TraditionalDetector();

    // 检测一帧，没检测到返回空（属正常情况）；debug_mask 输出分割掩膜，调阈值时看它。
    std::vector<ArmorDetection> detect(const cv::Mat& frame,
                                       cv::Mat* debug_mask = nullptr);

private:
    static LightBar makeLightBar(const cv::RotatedRect& rect);

    bool isLightBar(const LightBar& bar) const;

    // 给两根灯条打分，越小越像同一块板，负数表示不可能配对
    double pairCost(const LightBar& a, const LightBar& b) const;

    // 两根灯条的包围盒里夹着第三根时判为跨板配对（思路来自 rm_auto_aim 的 containLight）
    bool containLight(const LightBar& a, const LightBar& b,
                      const std::vector<LightBar>& all) const;

    double min_blue_dom_g_ = 45.0;   // B-G 下限
    double min_blue_dom_r_ = 60.0;   // B-R 下限
    double min_v_ = 25.0;            // 亮度下限，只滤全黑背景

    double min_red_dom_g_ = 50.0;    // 红方 R-G 下限，视频里的橙红灯条要它
    double min_red_dom_b_ = 80.0;    // 红方 R-B 下限

    // 过曝补光：主通道已饱和、另一通道滞后，用来捞回被打白的灯条
    double min_ov_sat_ = 240.0;      // 主通道饱和下限
    double min_ov_lag_ = 45.0;       // 另一通道滞后下限
    double min_ov_mid_ = 150.0;      // 绿通道下限，防单通道噪声尖峰

    double min_area_ = 15.0;         // 面积下限，滤噪点
    double max_area_ = 20000.0;      // 面积上限，视频近距离灯条约 4000px²，故放宽
    double min_aspect_ratio_ = 2.0;  // 长宽比下限
    double max_aspect_ratio_ = 60.0; // 长宽比上限

    // 两灯条中心距/灯条长 = 135mm/55mm，与距离无关，是配对主判据
    static constexpr double kLightBarDistRatio = 2.45;

    // 比值上限。偏航时中心距按 cosθ 缩短、灯条长不变，比值只减不增，超过 3.0 不可能同板
    static constexpr double kMaxBarDistRatio = 3.0;

    // 两灯条中心连线的最大倾角(度)
    static constexpr double kMaxCenterLineAngle = 30.0;
};
