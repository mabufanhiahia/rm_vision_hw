#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 结果展示：所有往图上画东西的代码都在这里，算法模块保持纯净。
class Visualizer {
public:
    // 画一块装甲板：四角框 + 角点 + 中心 + 数字/类别 + 置信度
    static void drawArmor(cv::Mat& image, const ArmorDetection& det,
                          const cv::Scalar& color = cv::Scalar(0, 255, 0));

    // 画真值位置：红色十字 + 一位编号。id_text 只放短串，长信息走 drawInfoBar
    static void drawTruth(cv::Mat& image, const cv::Point2f& px,
                          const std::string& id_text, int label_dy = -6);

    // 左上角信息条，那块通常是空背景，长文字放这里不会压住目标
    static void drawInfoBar(cv::Mat& image, const std::vector<std::string>& lines);

    // 替代直接 imshow：先把超屏的图等比缩小再建窗并显式设一次尺寸，
    // 否则窗口被顶到屏幕外，顶部（信息栏位置）会被裁掉。
    static void show(const std::string& window, const cv::Mat& image,
                     int max_w = 900, int max_h = 660);

private:
    // 线宽随框的大小自适应，避免远处小板被 2px 线糊满
    static int adaptiveThickness(const std::array<cv::Point2f, 4>& pts);

    // 带黑色描边的文字：画面上亮暗不一，纯色文字容易糊进背景
    static void putOutlinedText(cv::Mat& image, const std::string& text,
                                const cv::Point& org, double font_scale,
                                int thickness, const cv::Scalar& color);
};
