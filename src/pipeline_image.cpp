#include "pipeline.h"

#include "camera.h"
#include "detectors.h"
#include "ground_truth.h"
#include "pose_solver.h"
#include "visualizer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace {

constexpr int kLineWidth = 100;

// 检测与真值的最大匹配距离(mm)。实测三维误差在 55~371mm，留约 2.7 倍余量。
constexpr double kMatchMaxDistMm = 1000.0;
const std::string kLine(kLineWidth, '-');
const std::string kDoubleLine(kLineWidth, '=');

// 打印三维坐标，用 setw 对齐三列，方便上下两行比着看
void printVec3(std::ostream& os, double x, double y, double z) {
    os << std::fixed << std::setprecision(1)
       << "(" << std::setw(8) << x << ","
       << std::setw(8) << y << ","
       << std::setw(8) << z << ")";
}

std::string toFixed1(double v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << v;
    return ss.str();
}

std::string toFixed2(double v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << v;
    return ss.str();
}

// 传统方案绿色、神经网络青色，截图时一眼能看出是哪套算法出的
cv::Scalar colorOf(const std::string& m) {
    return (m == method::kNeural) ? cv::Scalar(255, 255, 0)   // 青色 (BGR)
                                  : cv::Scalar(0, 255, 0);    // 绿色
}

} // namespace

int runImage(const std::string& image_path, const std::string& gt_path,
             const std::string& config_path, const std::string& method_arg,
             const std::string& model_dir) {
    std::vector<std::string> methods;
    if (!parseMethods(method_arg, methods)) return -1;

    const CameraIntrinsics cam = camera::loadIntrinsics(config_path);
    if (!cam.valid) return -1;

    const std::string image_name = std::filesystem::path(image_path).stem().string();
    const std::vector<GroundTruth> gts =
        selectImage(loadGroundTruth(gt_path), image_name);
    if (gts.empty()) {
        std::cerr << "[警告] 真值表里没有 " << image_name
                  << " 的可见装甲板，将只做检测和解算，不做误差对比\n";
    }

    const cv::Mat origin = camera::loadImage(image_path);
    if (origin.empty()) return -1;

    DetectorSet detectors;
    detectors.init(model_dir, methods);
    // 图片一帧只跑一次，可以打开数字校验的逐块输出；视频模式不能开，会刷屏
    detectors.setVerbose(true);

    std::filesystem::create_directories("results");

    int ret = 0;

    for (const auto& m : methods) {
        std::cout << "\n" << kDoubleLine << "\n"
                  << "图片模式  方案 = " << m << "\n"
                  << "  图片: " << image_path
                  << "  (" << origin.cols << "x" << origin.rows << ")\n"
                  << "  内参: " << config_path;
        if (cam.image_size.width > 0) {
            std::cout << "  (标定分辨率 " << cam.image_size.width << "x"
                      << cam.image_size.height << ")";
        }
        std::cout << "\n" << kDoubleLine << "\n";

        cv::Mat debug_mask;
        const std::vector<ArmorDetection> detections =
            detectors.detect(origin, m, &debug_mask);
        if (!debug_mask.empty()) {
            cv::imwrite("results/debug_mask.jpg", debug_mask);
        }

        if (detections.empty()) {
            std::cout << "[结果] 未检测到装甲板。\n"
                      << "       排查建议：1) 打开 results/debug_mask.jpg 看灯条有没有被分割出来；\n"
                      << "                 2) 按需调 TraditionalDetector 的颜色/面积参数。\n";
            if (m == method::kNeural) {
                std::cout << "                 3) 神经网络方案请确认 models/armor.onnx 已正确下载。\n";
            }
            Visualizer::show("detect_result (no armor found) " + image_name, origin);
            cv::waitKey(0);
            cv::destroyAllWindows();
            continue;
        }

        // 把真值投影到图上，只标"真值该在哪"，不参与匹配
        std::vector<cv::Point3f> gt_pts_m;
        for (const auto& gt : gts) gt_pts_m.push_back(gt.t_cam);
        std::vector<cv::Point2f> gt_px;
        if (!gts.empty()) {
            // 零旋转零平移 = 真值坐标系就是相机坐标系，直接投影即可
            const cv::Mat r0 = cv::Mat::zeros(3, 1, CV_64F);
            const cv::Mat t0 = cv::Mat::zeros(3, 1, CV_64F);
            cv::projectPoints(gt_pts_m, r0, t0, cam.camera_matrix, cam.dist_coeffs, gt_px);
        }

        PoseSolver solver(cam.camera_matrix, cam.dist_coeffs,
                          PoseSolver::kDefaultArmorWidthMm,
                          PoseSolver::kDefaultArmorHeightMm);

        // 表头用 "|" 分隔而不是 setw：中文是双宽字符，setw 数出的列宽和显示宽度对不上
        std::cout << " 板# | 匹配真值 |"
                  << "        计算坐标(mm)          |"
                  << "        真值坐标(mm)          |"
                  << "    Δx |    Δy |    Δz |"
                  << " 三维误差 | 重投影px\n";
        std::cout << kLine << "\n";

        int solved = 0, matched = 0;
        double sum_err3d = 0.0, max_err3d = 0.0, sum_reproj = 0.0;
        std::map<int, std::string> err_text_of_gt; // 真值下标 → 误差文字，画图标注用

        std::vector<PoseResult> poses(detections.size());
        std::vector<bool> pose_ok(detections.size(), false);
        for (size_t k = 0; k < detections.size(); ++k) {
            poses[k] = solver.solve(detections[k]);
            pose_ok[k] = poses[k].success;
            if (pose_ok[k]) {
                ++solved;
                sum_reproj += poses[k].reproj_error;
            }
        }

        // 检测与真值一对一匹配：把所有组合按三维距离升序排，贪心占用。
        // 不然两块检测可能同时配到同一块真值，表格出现两行 armor_4，误差也被重复计入。
        std::vector<int> match_of_det(detections.size(), -1);
        if (!gts.empty() && solved > 0) {
            struct Pair { double dist; int det; int gt; };
            std::vector<Pair> pairs;
            for (size_t k = 0; k < detections.size(); ++k) {
                if (!pose_ok[k]) continue;
                for (size_t g = 0; g < gts.size(); ++g) {
                    const double dx = poses[k].tvec[0] - gts[g].t_cam.x * 1000.0;
                    const double dy = poses[k].tvec[1] - gts[g].t_cam.y * 1000.0;
                    const double dz = poses[k].tvec[2] - gts[g].t_cam.z * 1000.0;
                    const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (d <= kMatchMaxDistMm) pairs.push_back({d, static_cast<int>(k), static_cast<int>(g)});
                }
            }
            std::sort(pairs.begin(), pairs.end(),
                      [](const Pair& a, const Pair& b) { return a.dist < b.dist; });

            std::vector<bool> gt_used(gts.size(), false);
            for (const auto& p : pairs) {
                if (match_of_det[p.det] >= 0) continue; // 这块检测已配上
                if (gt_used[p.gt]) continue;            // 这块真值已被占用
                match_of_det[p.det] = p.gt;
                gt_used[p.gt] = true;
            }
        }

        for (size_t k = 0; k < detections.size(); ++k) {
            const auto& det = detections[k];
            const PoseResult& pose = poses[k];

            if (!pose_ok[k]) {
                // 解算失败也是结果的一部分，要说明原因
                const std::string why = (pose.reproj_error >= 0.0)
                    ? toFixed2(pose.reproj_error) + " px，超过阈值 " +
                      toFixed1(PoseSolver::kMaxReprojError) + " px"
                    : std::string("PnP 无解");
                std::cout << " " << std::setw(3) << (k + 1)
                          << " | 解算失败（" << why << "）\n";
                continue;
            }

            const int gt_index = match_of_det[k];
            std::cout << std::right << std::setw(4) << (k + 1) << " | ";
            if (gt_index >= 0) {
                ++matched;
                const GroundTruth& gt = gts[gt_index];
                const double gx = gt.t_cam.x * 1000.0;
                const double gy = gt.t_cam.y * 1000.0;
                const double gz = gt.t_cam.z * 1000.0;
                const double dx = pose.tvec[0] - gx;
                const double dy = pose.tvec[1] - gy;
                const double dz = pose.tvec[2] - gz;
                const double err3d = std::sqrt(dx * dx + dy * dy + dz * dz);
                sum_err3d += err3d;
                max_err3d = std::max(max_err3d, err3d);

                std::cout << std::setw(8)
                          << ("armor_" + std::to_string(gt.armor_id)) << " | ";
                printVec3(std::cout, pose.tvec[0], pose.tvec[1], pose.tvec[2]);
                std::cout << " | ";
                printVec3(std::cout, gx, gy, gz);
                std::cout << " | " << std::fixed << std::setprecision(1)
                          << std::setw(6) << dx << " | "
                          << std::setw(6) << dy << " | "
                          << std::setw(6) << dz << " | "
                          << std::setw(8) << err3d << " | "
                          << std::setw(8) << pose.reproj_error << "\n";

                err_text_of_gt[gt_index] = "err3d=" + toFixed1(err3d) + "mm";
            } else {
                std::cout << std::setw(8) << "-"
                          << " | ";
                printVec3(std::cout, pose.tvec[0], pose.tvec[1], pose.tvec[2]);
                std::cout << " | " << std::setw(28) << "-"
                          << " | " << std::setw(6) << "-"
                          << " | " << std::setw(6) << "-"
                          << " | " << std::setw(6) << "-"
                          << " | " << std::setw(8) << "-"
                          << " | " << std::setw(8) << pose.reproj_error
                          << "  （未匹配到真值）\n";
            }
        }

        std::cout << kLine << "\n";
        std::cout << " 检测 " << detections.size() << " 块 / 解算成功 " << solved
                  << " 块 / 匹配到真值 " << matched << " 块\n";
        if (matched > 0) {
            std::cout << " 平均三维误差 " << toFixed2(sum_err3d / matched)
                      << " mm | 最大 " << toFixed2(max_err3d) << " mm";
        }
        if (solved > 0) {
            std::cout << " | 平均重投影误差 " << toFixed2(sum_reproj / solved) << " px";
        }
        std::cout << "\n" << kDoubleLine << "\n";

        // 一块都没配上时直接把最可能的原因和下一步命令打出来，省得回头翻代码
        if (matched == 0 && solved > 0 && !gts.empty()) {
            std::cout << "[提示] 检测到了板，但与同组真值的三维距离全部超过 "
                      << static_cast<int>(kMatchMaxDistMm) << " mm。\n"
                      << "       最常见原因：真值表的 image 列与图片文件编号错位一格\n"
                      << "       （判定依据见 data/gt/README.md）。\n"
                      << "       默认用的是重排后的表，可以换原表对比一次：\n"
                      << "         ./build/rm_vision_hw image " << image_path
                      << " data/gt/ground_truth.csv\n\n";
        }

        // 画标注图
        cv::Mat canvas = origin.clone();
        const cv::Scalar color = colorOf(m);
        for (const auto& det : detections) {
            Visualizer::drawArmor(canvas, det, color);
        }
        // 真值点全画出来，没被检测到的会看得出来；相邻点按奇偶上下错开避免编号重叠
        for (size_t i = 0; i < gts.size() && i < gt_px.size(); ++i) {
            Visualizer::drawTruth(canvas, gt_px[i], std::to_string(gts[i].armor_id),
                                  (i % 2 == 0) ? -6 : 14);
        }
        // 长文字统一放左上角信息栏，写在目标旁边会直接盖住装甲板
        std::vector<std::string> info = {
            "method: " + m,
            "image: " + image_name,
            "cross = truth position, box = detection",
        };
        for (size_t i = 0; i < gts.size(); ++i) {
            auto it = err_text_of_gt.find(static_cast<int>(i));
            info.push_back("  armor_" + std::to_string(gts[i].armor_id) + "  " +
                           ((it != err_text_of_gt.end()) ? it->second
                                                         : std::string("(no detection)")));
        }
        Visualizer::drawInfoBar(canvas, info);

        const std::string out_path =
            "results/detect_" + image_name + "_" + m + ".jpg";
        cv::imwrite(out_path, canvas);

        // WSL 里 imshow 走 WSLg，报 "cannot open display" 时看保存的文件版即可
        Visualizer::show("detect_result [" + m + "] " + image_name, canvas);
        cv::waitKey(0);
        cv::destroyAllWindows();

        std::cout << "标注图已保存: " << out_path << "\n";
    }

    return ret;
}
