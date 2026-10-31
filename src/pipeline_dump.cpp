#include "pipeline.h"

#include "camera.h"
#include "detectors.h"
#include "frame_pose.h"
#include "imu_reader.h"
#include "pose_solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

// 作业给的 IMU 时间偏移：文件里的时间戳是"收到时刻"，采集时刻要往前退 9ms
constexpr double kImuOffsetSec = 0.009;

// 帧间把同一块板配起来的距离上限。车自转时板一帧最多走几厘米，0.6m 足够宽松
constexpr double kMatchRadiusM = 0.6;

// 打分时"半径要接近官方值"这一项的权重。主判据还是高度稳定性，
// 半径只用来在几个高度稳定性差不多的候选之间做取舍，所以权重给小一点。
constexpr double kRadiusWeight = 0.5;

struct FrameRecord {
    int frame = 0;
    double imu_ts = 0.0;
    double capture_ts = 0.0;
    std::vector<ArmorPoseCam> armors;
};

double mean(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double stdev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double s = 0.0;
    for (const double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / v.size());
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// 把一堆平面点拟合成圆：x²+y² = 2ax + 2by + c，最小二乘解一次就行
bool fitCircle(const std::vector<cv::Vec2d>& pts, cv::Vec2d& center, double& radius) {
    const int n = static_cast<int>(pts.size());
    if (n < 4) return false;

    cv::Mat A(n, 3, CV_64F);
    cv::Mat b(n, 1, CV_64F);
    for (int i = 0; i < n; ++i) {
        const double x = pts[i][0], y = pts[i][1];
        A.at<double>(i, 0) = 2 * x;
        A.at<double>(i, 1) = 2 * y;
        A.at<double>(i, 2) = 1.0;
        b.at<double>(i) = x * x + y * y;
    }

    cv::Mat sol;
    if (!cv::solve(A, b, sol, cv::DECOMP_SVD)) return false;

    const double a = sol.at<double>(0), bb = sol.at<double>(1), c = sol.at<double>(2);
    const double r2 = c + a * a + bb * bb;
    if (r2 <= 0.0) return false;

    center = cv::Vec2d(a, bb);
    radius = std::sqrt(r2);
    return true;
}

// 一个候选外参的打分结果
struct Score {
    int index = 0;
    std::string name;
    double height_std = 1e9; // 主判据：世界系下板高度的标准差，车不跳所以它必须小
    double radius = 0.0;     // 拟合出来的半径，拿来和官方值对一对
    double circle_rms = 1e9; // 所有板到拟合圆的残差，车中心不动时才该很小
    double step_median = 1e9;// 帧间同一块板的位移中位数
    double cost = 1e9;       // 最终排序用
    cv::Vec2d center = cv::Vec2d(0, 0);
};

// 判据：目标车不会上下跳，所以世界系下板的高度必须恒定；外参错了云台转动会漏进高度。
// 半径只用来在高度稳定性接近的候选之间取舍。
Score scoreExtrinsic(const std::vector<FrameRecord>& frames,
                     const std::vector<ImuSample>& imu,
                     const ExtrinsicOption& opt, int index) {
    std::vector<cv::Vec2d> pts;
    std::vector<double> heights;
    std::vector<std::vector<cv::Vec3d>> world_per_frame;

    for (const auto& f : frames) {
        if (f.frame - 1 >= static_cast<int>(imu.size())) break;
        const cv::Mat R_wc = worldRotation(imu[f.frame - 1].quat, opt);

        std::vector<cv::Vec3d> cur;
        for (const auto& a : f.armors) {
            const cv::Vec3d w = camToWorld(a.pos, R_wc);
            cur.push_back(w);
            pts.emplace_back(w[0], w[1]);
            heights.push_back(w[2]);
        }
        world_per_frame.push_back(cur);
    }

    Score s;
    s.index = index;
    s.name = opt.name;
    s.height_std = stdev(heights);
    if (pts.size() < 4) return s;

    cv::Vec2d center;
    double radius = 0.0;
    if (!fitCircle(pts, center, radius)) return s;
    s.center = center;
    s.radius = radius;

    double sum = 0.0;
    for (const auto& p : pts) {
        const double d = std::hypot(p[0] - center[0], p[1] - center[1]) - radius;
        sum += d * d;
    }
    s.circle_rms = std::sqrt(sum / pts.size());

    // 副判据：相邻帧把最近的板配起来，看世界系下它一帧挪了多少
    std::vector<double> steps;
    for (size_t k = 1; k < world_per_frame.size(); ++k) {
        for (const auto& cur : world_per_frame[k]) {
            double best = kMatchRadiusM;
            for (const auto& prev : world_per_frame[k - 1]) {
                best = std::min(best, cv::norm(cur - prev));
            }
            if (best < kMatchRadiusM) steps.push_back(best);
        }
    }
    s.step_median = median(steps);
    s.cost = s.height_std + kRadiusWeight * std::fabs(s.radius - armor_geom::kRadiusMeanM);
    return s;
}

} // namespace

int runDump(const std::string& video_path, const std::string& imu_path,
            const std::string& config_path, const std::string& method_arg,
            const std::string& model_dir, const std::string& csv_path,
            const std::string& extrinsic_arg) {
    std::vector<std::string> methods;
    if (!parseMethods(method_arg, methods)) return -1;
    if (methods.size() > 1) {
        std::cout << "[提示] dump 模式只用第一种方案（" << methods[0] << "）取位姿。\n";
        methods.resize(1);
    }
    const std::string method = methods[0];

    // 内参必须用视频那套（1440×1080 带畸变），用图片那套解出来会整体偏
    const CameraIntrinsics intr = camera::loadIntrinsics(config_path);
    if (!intr.valid) {
        std::cerr << "[错误] 读不到内参: " << config_path << "\n";
        return -1;
    }

    std::vector<ImuSample> imu;
    std::string err;
    if (!ImuReader::load(imu_path, kImuOffsetSec, imu, &err)) {
        std::cerr << "[错误] " << err << "\n";
        return -1;
    }
    std::cout << "========== dump 模式 ==========\n"
              << "视频: " << video_path << "\n"
              << "IMU : " << imu_path << "  " << imu.size() << " 行"
              << "（采集时刻已退 " << kImuOffsetSec * 1000 << " ms）\n"
              << "内参: " << config_path << "\n"
              << "方案: " << method << "\n";

    cv::VideoCapture cap;
    double fps = 30.0;
    int total_frames = 0;
    if (!camera::openVideo(video_path, cap, &fps, &total_frames)) return -1;

    DetectorSet detectors;
    if (!detectors.init(model_dir, methods)) {
        std::cerr << "[警告] 模型没加载全，可能什么都检不出来。\n";
    }

    PoseSolver solver(intr.camera_matrix, intr.dist_coeffs,
                      PoseSolver::kDefaultArmorWidthMm, PoseSolver::kDefaultArmorHeightMm);

    std::vector<FrameRecord> frames;
    cv::Mat frame;
    int frame_index = 0;
    long long detected_total = 0;
    long long solved_total = 0;
    const auto t_start = std::chrono::steady_clock::now();

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;
        ++frame_index;

        FrameRecord rec;
        rec.frame = frame_index;
        if (frame_index - 1 < static_cast<int>(imu.size())) {
            rec.imu_ts = imu[frame_index - 1].timestamp;
            rec.capture_ts = imu[frame_index - 1].capture_time;
        }

        const std::vector<ArmorDetection> dets = detectors.detect(frame, method);
        detected_total += static_cast<long long>(dets.size());

        for (const auto& d : dets) {
            const PoseResult p = solver.solve(d);
            if (!p.success) continue; // 重投影误差超限的丢掉，别污染拟合

            ArmorPoseCam a;
            a.frame = frame_index;
            a.digit = d.digit;
            a.pos = p.tvec / 1000.0; // PnP 出来是毫米，统一成米
            a.rvec = p.rvec;
            a.reproj = p.reproj_error;
            a.image_points = d.image_points;
            rec.armors.push_back(a);
        }
        solved_total += static_cast<long long>(rec.armors.size());
        frames.push_back(rec);

        if (frame_index % 60 == 0) {
            std::cout << "  已处理 " << frame_index << " 帧，本帧解算 "
                      << rec.armors.size() << " 块" << std::endl;
        }
    }
    cap.release();

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cout << "处理完 " << frames.size() << " 帧，耗时 " << std::fixed << std::setprecision(1)
              << elapsed << " s；检出 " << detected_total << " 块、解算成功 "
              << solved_total << " 块\n";

    if (solved_total < 20) {
        std::cerr << "[错误] 解算成功的板太少（" << solved_total
                  << " 块），判不了外参。先确认内参是不是视频那套、识别方案能不能检出来。\n";
        return -1;
    }

    // ---- 标定外参 ----
    const std::vector<ExtrinsicOption> opts = enumerateExtrinsics();
    std::vector<Score> scores;
    for (size_t i = 0; i < opts.size(); ++i) {
        scores.push_back(scoreExtrinsic(frames, imu, opts[i], static_cast<int>(i)));
    }

    int best = -1;
    if (!extrinsic_arg.empty()) {
        // 也可以手动指定编号，跳过自动选（自动选错了时用）
        best = std::stoi(extrinsic_arg);
        if (best < 0 || best >= static_cast<int>(opts.size())) {
            std::cerr << "[错误] 外参编号要在 0~" << opts.size() - 1 << " 之间\n";
            return -1;
        }
    } else {
        for (size_t i = 0; i < scores.size(); ++i) {
            if (best < 0 || scores[i].cost < scores[best].cost) best = static_cast<int>(i);
        }
    }

    std::cout << "\n---- 外参候选打分（主判据：世界系板高度标准差，越小越对）----\n";
    std::vector<int> order(scores.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a].cost < scores[b].cost;
    });
    std::cout << std::fixed << std::setprecision(4);
    for (int k = 0; k < std::min<size_t>(5, order.size()); ++k) {
        const Score& s = scores[order[k]];
        std::cout << "  #" << s.index << "  " << s.name
                  << "\n      高度std " << s.height_std << " m"
                  << "   半径 " << s.radius << " m"
                  << "   圆残差 " << s.circle_rms << " m"
                  << "   帧间位移中位 " << s.step_median << " m\n";
    }

    const Score& bs = scores[best];
    std::cout << "\n选中外参 #" << best << "  " << bs.name << "\n"
              << "  拟合圆心（车中心）= (" << bs.center[0] << ", " << bs.center[1] << ") m\n"
              << "  拟合半径 = " << bs.radius << " m"
              << "（官方 /tf 真值 " << armor_geom::kRadiusMeanM << " m）\n"
              << "  板高度标准差 = " << bs.height_std << " m（车不跳，这项应该只有几厘米）\n";

    // ---- 导出 CSV ----
    std::ofstream out(csv_path);
    if (!out) {
        std::cerr << "[错误] 写不了 CSV: " << csv_path << "\n";
        return -1;
    }
    out << "frame,imu_ts,capture_ts,idx,digit,"
        << "x_cam,y_cam,z_cam,reproj,x_w,y_w,z_w,"
        << "p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y\n";
    out << std::fixed << std::setprecision(6);

    for (const auto& f : frames) {
        if (f.frame - 1 >= static_cast<int>(imu.size())) break;
        const cv::Mat R_wc = worldRotation(imu[f.frame - 1].quat, opts[best]);
        for (size_t i = 0; i < f.armors.size(); ++i) {
            const ArmorPoseCam& a = f.armors[i];
            const cv::Vec3d w = camToWorld(a.pos, R_wc);
            out << f.frame << "," << f.imu_ts << "," << f.capture_ts << "," << i << "," << a.digit
                << "," << a.pos[0] << "," << a.pos[1] << "," << a.pos[2]
                << "," << a.reproj
                << "," << w[0] << "," << w[1] << "," << w[2];
            for (const auto& p : a.image_points) {
                out << "," << p.x << "," << p.y;
            }
            out << "\n";
        }
    }
    out.close();
    std::cout << "位姿已导出: " << csv_path << "\n";
    std::cout << "===============================\n";
    return 0;
}
