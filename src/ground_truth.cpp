#include "ground_truth.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

std::vector<GroundTruth> loadGroundTruth(const std::string& csv_path) {
    std::vector<GroundTruth> table;

    std::ifstream fin(csv_path);
    if (!fin.is_open()) {
        std::cerr << "[警告] 打不开真值表: " << csv_path
                  << "，将跳过误差对比，只做检测和解算\n";
        return table;
    }

    std::string line;
    int line_no = 0;
    while (std::getline(fin, line)) {
        ++line_no;

        // 剥 UTF-8 BOM（记事本保存的文件开头有 3 个隐藏字节），不处理表头识别会失效
        if (line_no == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            line.erase(0, 3);
        }
        if (!line.empty() && line.back() == '\r') line.pop_back(); // 剥 Windows 的 CRLF 行尾

        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("image,", 0) == 0) continue;

        try {
            std::stringstream ss(line);
            std::string cell;
            GroundTruth gt;
            std::getline(ss, gt.image, ',');
            std::getline(ss, cell, ','); gt.armor_id = std::stoi(cell);
            std::getline(ss, cell, ','); gt.visible = (cell == "front");
            std::getline(ss, cell, ','); gt.t_cam.x = static_cast<float>(std::stod(cell));
            std::getline(ss, cell, ','); gt.t_cam.y = static_cast<float>(std::stod(cell));
            std::getline(ss, cell, ','); gt.t_cam.z = static_cast<float>(std::stod(cell));
            table.push_back(gt);
        } catch (const std::exception& e) {
            // 一行坏了只跳过这一行，不让整个程序挂掉
            std::cerr << "[警告] 真值表第 " << line_no << " 行解析失败，已跳过: "
                      << e.what() << "\n";
        }
    }
    return table;
}

std::vector<GroundTruth> selectImage(const std::vector<GroundTruth>& all,
                                     const std::string& image_name) {
    std::vector<GroundTruth> out;
    for (const auto& gt : all) {
        if (gt.image == image_name && gt.visible) out.push_back(gt);
    }
    return out;
}

const GroundTruth* matchBy3DDistance(const cv::Vec3d& calc_mm,
                                     const std::vector<GroundTruth>& gts,
                                     double max_dist_mm,
                                     double* dist_out) {
    const GroundTruth* best = nullptr;
    double best_dist = 0.0;

    for (const auto& gt : gts) {
        // 真值是米，乘 1000 换成毫米才能与 calc_mm 直接相减
        const double dx = calc_mm[0] - static_cast<double>(gt.t_cam.x) * 1000.0;
        const double dy = calc_mm[1] - static_cast<double>(gt.t_cam.y) * 1000.0;
        const double dz = calc_mm[2] - static_cast<double>(gt.t_cam.z) * 1000.0;
        const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (best == nullptr || d < best_dist) {
            best_dist = d;
            best = &gt;
        }
    }

    // 离最近的真值都远超合理范围，多半是误检，宁可报"未匹配"也不硬凑一个数污染统计
    if (best != nullptr && best_dist > max_dist_mm) {
        best = nullptr;
    }
    if (dist_out != nullptr) *dist_out = best_dist;
    return best;
}
