#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace {

// CSV 读完放进这几个向量，画图只认它们
struct Series {
    std::vector<double> t;
    std::vector<double> cx, vx, cy, vy, za, vz, theta, omega, r;
    // 9 个状态量各自的 1 倍标准差，顺序与上面一致，来自 ekf 模式的 CSV 末尾 9 列
    std::vector<std::vector<double>> sd;
    bool has_sd = false;
};

bool loadStates(const std::string& path, Series& s, std::string& err) {
    std::ifstream in(path);
    if (!in) {
        err = "打不开状态文件: " + path;
        return false;
    }

    std::string line;
    if (!std::getline(in, line)) {
        err = "状态文件是空的";
        return false;
    }
    s.sd.assign(9, std::vector<double>());
    size_t sd_rows = 0;
    // 表头形如 frame,t,cx,vx,...,resid,inited，这里按列号取，够用且不依赖列序
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<double> col;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try {
                col.push_back(std::stod(cell));
            } catch (...) {
                col.push_back(0.0);
            }
        }
        if (col.size() < 11) continue;
        s.t.push_back(col[1]);
        s.cx.push_back(col[2]);
        s.vx.push_back(col[3]);
        s.cy.push_back(col[4]);
        s.vy.push_back(col[5]);
        s.za.push_back(col[6]);
        s.vz.push_back(col[7]);
        s.theta.push_back(col[8]);
        s.omega.push_back(col[9]);
        s.r.push_back(col[10]);

        // 老版本的 CSV 没有 sigma 列，缺了就按 0 填，画出来就是不带阴影的普通曲线
        if (col.size() >= 23) {
            for (int k = 0; k < 9; ++k) s.sd[k].push_back(col[14 + k]);
            ++sd_rows;
        } else {
            for (int k = 0; k < 9; ++k) s.sd[k].push_back(0.0);
        }
    }
    s.has_sd = (sd_rows == s.t.size() && sd_rows > 0);
    if (s.t.size() < 10) {
        err = "有效数据不到 10 行，画不出曲线";
        return false;
    }
    return true;
}

// theta 在 [-pi, pi) 之间折回过，画曲线前要解缠，否则会出现从 +180 跳到 -180 的竖线
std::vector<double> unwrapDeg(const std::vector<double>& rad) {
    std::vector<double> out;
    out.reserve(rad.size());
    double offset = 0.0;
    double prev = 0.0;
    for (size_t i = 0; i < rad.size(); ++i) {
        double cur = rad[i] * 180.0 / CV_PI;
        if (i > 0) {
            double d = cur + offset - prev;
            while (d > 180.0) { offset -= 360.0; d -= 360.0; }
            while (d < -180.0) { offset += 360.0; d += 360.0; }
        }
        prev = cur + offset;
        out.push_back(prev);
    }
    return out;
}

void drawOne(cv::Mat& canvas, cv::Rect box,
             const std::vector<double>& xs, const std::vector<double>& ys,
             const std::vector<double>& sd,
             const std::string& title, const std::string& unit,
             const cv::Scalar& color) {
    const int left = box.x + 56;
    const int right = box.x + box.width - 14;
    const int top = box.y + 30;
    const int bottom = box.y + box.height - 34;

    double x0 = xs.front(), x1 = xs.back();
    if (x1 - x0 < 1e-9) x1 = x0 + 1.0;
    double ymin = *std::min_element(ys.begin(), ys.end());
    double ymax = *std::max_element(ys.begin(), ys.end());
    const bool with_band = (sd.size() == ys.size());
    if (with_band) {
        for (size_t i = 0; i < ys.size(); ++i) {
            ymin = std::min(ymin, ys[i] - sd[i]);
            ymax = std::max(ymax, ys[i] + sd[i]);
        }
    }
    if (ymax - ymin < 1e-9) { ymin -= 0.5; ymax += 0.5; }
    const double pad = (ymax - ymin) * 0.08;
    ymin -= pad;
    ymax += pad;

    auto px = [&](double x) { return left + (x - x0) / (x1 - x0) * (right - left); };
    auto py = [&](double y) { return bottom - (y - ymin) / (ymax - ymin) * (bottom - top); };

    // 边框 + 网格
    cv::rectangle(canvas, cv::Point(left, top), cv::Point(right, bottom),
                  cv::Scalar(200, 200, 200), 1);
    for (int i = 1; i < 4; ++i) {
        const int y = top + (bottom - top) * i / 4;
        cv::line(canvas, cv::Point(left, y), cv::Point(right, y), cv::Scalar(232, 232, 232), 1);
    }

    // ±1sigma 不确定度带。它描的是滤波器自己认为的可信范围：
    // 带子收窄说明滤波器越来越自信，真实曲线若常跑出带外，说明协方差给乐观了
    if (with_band) {
        std::vector<cv::Point> band;
        band.reserve(2 * ys.size());
        for (size_t i = 0; i < ys.size(); ++i)
            band.emplace_back(cvRound(px(xs[i])), cvRound(py(ys[i] + sd[i])));
        for (size_t i = ys.size(); i-- > 0;)
            band.emplace_back(cvRound(px(xs[i])), cvRound(py(ys[i] - sd[i])));
        const cv::Scalar light(color[0] * 0.35 + 255 * 0.65,
                               color[1] * 0.35 + 255 * 0.65,
                               color[2] * 0.35 + 255 * 0.65);
        cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{band}, light, cv::LINE_AA);
    }

    // 曲线
    std::vector<cv::Point> pts;
    pts.reserve(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        pts.emplace_back(cvRound(px(xs[i])), cvRound(py(ys[i])));
    }
    cv::polylines(canvas, std::vector<std::vector<cv::Point>>{pts}, false, color, 1,
                  cv::LINE_AA);

    // 标题、单位、量程
    cv::putText(canvas, title, cv::Point(box.x + 12, box.y + 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    cv::putText(canvas, unit, cv::Point(right - 6 - 8 * static_cast<int>(unit.size()), top - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(110, 110, 110), 1, cv::LINE_AA);

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f", ymax);
    cv::putText(canvas, buf, cv::Point(box.x + 4, top + 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "%.3f", ymin);
    cv::putText(canvas, buf, cv::Point(box.x + 4, bottom + 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);

    if (with_band) {
        std::snprintf(buf, sizeof(buf), "1s %.3f", sd.back());
        cv::putText(canvas, buf, cv::Point(box.x + 4, top + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
    }

    std::snprintf(buf, sizeof(buf), "%.1fs", x0);
    cv::putText(canvas, buf, cv::Point(left, bottom + 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
    std::snprintf(buf, sizeof(buf), "%.1fs", x1);
    cv::putText(canvas, buf, cv::Point(right - 34, bottom + 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
}

} // namespace

int runPlot(const std::string& csv_path, const std::string& out_path) {
    Series s;
    std::string err;
    if (!loadStates(csv_path, s, err)) {
        std::cerr << "[错误] " << err << "\n";
        return -1;
    }

    const std::vector<double> theta_deg = unwrapDeg(s.theta);

    const int cell_w = 430, cell_h = 280, pad = 50;
    const int cols = 3, rows = 3;
    cv::Mat canvas(rows * cell_h + 2 * pad, cols * cell_w + 2 * pad, CV_8UC3,
                   cv::Scalar(255, 255, 255));

    struct Item {
        const std::vector<double>* ys;
        const std::vector<double>* sd;
        std::string title;
        std::string unit;
        cv::Scalar color;
    };

    // 没有 sigma 列时传空向量，drawOne 会自动退化成不带阴影的曲线
    const std::vector<double> kEmpty;
    auto sdv = [&](int k) -> const std::vector<double>& {
        return s.has_sd ? s.sd[k] : kEmpty;
    };
    std::vector<double> sd_theta_deg;
    if (s.has_sd) {
        for (double v : s.sd[6]) sd_theta_deg.push_back(v * 180.0 / CV_PI);
    }
    const cv::Scalar blue(180, 110, 30);
    const cv::Scalar green(90, 150, 40);
    const cv::Scalar red(50, 60, 200);

    const std::vector<Item> items = {
        {&s.cx,    &sdv(0), "c_x",   "m",     blue},
        {&s.vx,    &sdv(1), "v_x",   "m/s",   green},
        {&theta_deg, s.has_sd ? &sd_theta_deg : &kEmpty, "theta", "deg", red},
        {&s.cy,    &sdv(2), "c_y",   "m",     blue},
        {&s.vy,    &sdv(3), "v_y",   "m/s",   green},
        {&s.omega, &sdv(7), "omega", "rad/s", red},
        {&s.za,    &sdv(4), "z_a",   "m",     blue},
        {&s.vz,    &sdv(5), "v_z",   "m/s",   green},
        {&s.r,     &sdv(8), "r",     "m",     red},
    };

    for (size_t i = 0; i < items.size(); ++i) {
        const int c = static_cast<int>(i) % cols;
        const int r = static_cast<int>(i) / cols;
        const cv::Rect box(pad + c * cell_w, pad + r * cell_h, cell_w, cell_h);
        drawOne(canvas, box, s.t, *items[i].ys, *items[i].sd,
                items[i].title, items[i].unit, items[i].color);
    }

    cv::putText(canvas, "shaded band = +-1 sigma from the EKF covariance",
                cv::Point(pad, rows * cell_h + 2 * pad - 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(110, 110, 110), 1, cv::LINE_AA);

    if (!cv::imwrite(out_path, canvas)) {
        std::cerr << "[错误] 图片写不进去: " << out_path << "\n";
        return -1;
    }

    std::cout << "曲线图已保存: " << out_path << "\n"
              << "  共 " << s.t.size() << " 个状态采样，"
              << "时间 " << s.t.front() << " ~ " << s.t.back() << " s\n"
              << "  不确定度带: " << (s.has_sd ? "有（读到了 sigma 列）" : "无（CSV 里没有 sigma 列）")
              << "\n";
    return 0;
}
