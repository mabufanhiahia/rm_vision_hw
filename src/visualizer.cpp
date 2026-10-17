#include "visualizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

int Visualizer::adaptiveThickness(const std::array<cv::Point2f, 4>& pts) {
    double perimeter = 0.0;
    for (int i = 0; i < 4; ++i) {
        perimeter += cv::norm(pts[i] - pts[(i + 1) % 4]);
    }
    const int t = static_cast<int>(std::lround(perimeter / 300.0));
    return std::min(4, std::max(1, t)); // 夹到 [1,4]，远处小板不会被粗线糊满
}

void Visualizer::putOutlinedText(cv::Mat& image, const std::string& text,
                                 const cv::Point& org, double font_scale,
                                 int thickness, const cv::Scalar& color) {
    // 先描一圈黑边再写彩色正文，否则文字容易糊进亮背景
    cv::putText(image, text, org, cv::FONT_HERSHEY_SIMPLEX, font_scale,
                cv::Scalar(0, 0, 0), thickness + 2, cv::LINE_AA);
    cv::putText(image, text, org, cv::FONT_HERSHEY_SIMPLEX, font_scale,
                color, thickness, cv::LINE_AA);
}

void Visualizer::drawArmor(cv::Mat& image, const ArmorDetection& det,
                           const cv::Scalar& color) {
    if (image.empty()) return;

    const int thickness = adaptiveThickness(det.image_points);

    // 四条边：左上→右上→右下→左下→回到左上
    for (int i = 0; i < 4; ++i) {
        cv::line(image, det.image_points[i], det.image_points[(i + 1) % 4],
                 color, thickness);
    }

    // 角点，便于判断框歪没歪、角点有没有落在灯条端点上
    for (const auto& p : det.image_points) {
        cv::circle(image, p, thickness, cv::Scalar(0, 255, 255), -1);
    }

    cv::Point2f center(0.f, 0.f);
    float min_y = det.image_points[0].y, max_y = det.image_points[0].y;
    for (const auto& p : det.image_points) {
        center += p;
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    center *= 0.25f;
    cv::circle(image, center, thickness + 1, color, -1);

    // 字号跟着框的对角线缩放，小板上不压框、大板上看得清
    const double diag = cv::norm(det.image_points[0] - det.image_points[2]);
    const double font_scale = std::min(0.9, std::max(0.35, diag / 160.0));
    const int font_thickness = std::max(1, thickness);

    std::string text;
    if (det.digit > 0) {
        text = std::to_string(det.digit);   // 认出数字就显示数字
    } else if (!det.class_name.empty()) {
        text = det.class_name;              // 神经网络方案显示兵种
    }
    if (det.confidence > 0.f) {             // 传统方案默认 0，不显示
        char buf[32];
        std::snprintf(buf, sizeof(buf), " %.2f", det.confidence);
        text += buf;
    }

    if (!text.empty()) {
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                            font_scale, font_thickness, &baseline);
        // 标签默认在框正上方，顶出画面就翻到下方；左右也夹一下避免被截断
        cv::Point org(cvRound(center.x - ts.width / 2.0), cvRound(min_y - 6));
        if (org.y - ts.height < 0) {
            org.y = cvRound(max_y + ts.height + 6);
        }
        org.x = std::min(std::max(org.x, 2), std::max(2, image.cols - ts.width - 2));
        org.y = std::min(std::max(org.y, ts.height + 2), std::max(2, image.rows - 2));
        putOutlinedText(image, text, org, font_scale, font_thickness, color);
    }
}

void Visualizer::drawTruth(cv::Mat& image, const cv::Point2f& px,
                           const std::string& id_text, int label_dy) {
    if (image.empty()) return;
    const cv::Point p(cvRound(px.x), cvRound(px.y));

    // 十字用 1px：4m 处板宽才 30px，粗十字会把整块板遮掉
    cv::drawMarker(image, p, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 12, 1);
    cv::circle(image, p, 2, cv::Scalar(0, 0, 255), -1);

    if (!id_text.empty()) {
        putOutlinedText(image, id_text, p + cv::Point(7, label_dy), 0.5, 1,
                        cv::Scalar(0, 0, 255));
    }
}

void Visualizer::drawInfoBar(cv::Mat& image, const std::vector<std::string>& lines) {
    if (image.empty() || lines.empty()) return;

    // 半透明黑底：ROI 与纯黑图按权重混合，比实心黑块柔和
    const int line_h = 22;
    const int pad = 6;
    const int h = static_cast<int>(lines.size()) * line_h + pad * 2;
    int w = 0;
    for (const auto& s : lines) {
        int baseline = 0;
        w = std::max(w, cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline).width);
    }
    w += pad * 2;
    w = std::min(w, image.cols); // 画面小就按宽度截断，不越界
    const cv::Rect bar(0, 0, w, std::min(h, image.rows));
    cv::Mat roi = image(bar);
    cv::Mat dark(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(dark, 0.45, roi, 0.55, 0.0, roi);

    for (size_t i = 0; i < lines.size(); ++i) {
        const cv::Point org(pad, pad + line_h * static_cast<int>(i) + 14);
        if (org.y >= image.rows) break;
        putOutlinedText(image, lines[i], org, 0.5, 1, cv::Scalar(255, 255, 255));
    }
}

void Visualizer::show(const std::string& window, const cv::Mat& image,
                      int max_w, int max_h) {
    if (image.empty()) return;

    // 超屏就等比缩小一份显示副本；resize 另开 Mat，imwrite 存的仍是原分辨率
    cv::Mat shown = image;
    if (image.cols > max_w || image.rows > max_h) {
        const double s = std::min(static_cast<double>(max_w) / image.cols,
                                  static_cast<double>(max_h) / image.rows);
        cv::resize(image, shown, cv::Size(), s, s, cv::INTER_AREA);
    }

    // 尺寸只在第一次设，否则视频每帧都会把窗口拉回默认大小
    static std::set<std::string> created;
    if (created.find(window) == created.end()) {
        cv::namedWindow(window, cv::WINDOW_NORMAL);
        // WINDOW_NORMAL 建出来是几百像素的小窗，必须显式设一次尺寸
        cv::resizeWindow(window, shown.cols, shown.rows);
        created.insert(window);
    }
    cv::imshow(window, shown);
}
