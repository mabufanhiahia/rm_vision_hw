#include "pipeline.h"

#include "camera.h"
#include "detectors.h"
#include "ekf.h"
#include "frame_pose.h"
#include "imu_reader.h"
#include "pose_solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <opencv2/highgui.hpp>

namespace {

// 作业给的 IMU 时间偏移：文件里是"收到时刻"，采集时刻要往前退 9ms
constexpr double kImuOffsetSec = 0.009;

// dump 标定出来的相机-IMU 安装外参编号，见 include/frame_pose.h 的候选表
constexpr int kExtrinsicIndex = 17;

// 启动阶段用前 60 帧的板位拟合圆心，拟合完才开始输出状态
constexpr int kBootstrapFrames = 60;

// 板号匹配用角度差，不用位置距离（理由见 ekf.h 里 matchByAngle 的注释）
constexpr double kMatchAngleRad = 75.0 * CV_PI / 180.0;
// 跳变重定之后放宽到 120°，此时 theta 已经按观测改过，能容忍一整块板的切换
constexpr double kReacquireAngleRad = 120.0 * CV_PI / 180.0;

struct FramePoses {
    int frame = 0;
    double capture_ts = 0.0;
    std::vector<ArmorPoseCam> armors;
};

double meanOf(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// 平面圆最小二乘：x²+y² = 2ax + 2by + c
bool fitCircle(const std::vector<cv::Vec2d>& pts, cv::Vec2d& center, double& radius) {
    const int n = static_cast<int>(pts.size());
    if (n < 8) return false;

    cv::Mat A(n, 3, CV_64F), b(n, 1, CV_64F);
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

// 世界系点 → 相机系。R_wc 满足 p_w = R_wc · p_c，所以反变换只要转置
cv::Vec3d worldToCam(const cv::Vec3d& p_w, const cv::Mat& R_wc) {
    return cv::Vec3d(R_wc.at<double>(0, 0) * p_w[0] + R_wc.at<double>(1, 0) * p_w[1] + R_wc.at<double>(2, 0) * p_w[2],
                     R_wc.at<double>(0, 1) * p_w[0] + R_wc.at<double>(1, 1) * p_w[1] + R_wc.at<double>(2, 1) * p_w[2],
                     R_wc.at<double>(0, 2) * p_w[0] + R_wc.at<double>(1, 2) * p_w[1] + R_wc.at<double>(2, 2) * p_w[2]);
}

// 画线和写字要整数像素，投影出来的是 double
cv::Point toInt(const cv::Point2d& p) { return cv::Point(cvRound(p.x), cvRound(p.y)); }

// 世界系点投到像素。输出容器必须也是 Point2d——projectPoints 按输入深度创建输出，
// 给 Point2f 会撞 "mtype == type0" 断言
std::vector<cv::Point2d> projectWorld(const std::vector<cv::Point3d>& pts,
                                      const cv::Mat& R_wc,
                                      const cv::Mat& K, const cv::Mat& D) {
    std::vector<cv::Point3d> obj;
    obj.reserve(pts.size());
    for (const auto& p : pts) {
        const cv::Vec3d c = worldToCam(cv::Vec3d(p.x, p.y, p.z), R_wc);
        obj.emplace_back(c[0], c[1], c[2]);
    }
    std::vector<cv::Point2d> img;
    cv::projectPoints(obj, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), K, D, img);
    return img;
}

// 俯视窗口：世界系 x-y 平面。画面里云台一直在动，看不出车中心到底有没有漂，
// 这个窗口能直接看到车中心轨迹，以及四块板是不是真的绕着它转
class TopView {
public:
    TopView(const cv::Vec2d& origin, double half_span_m = 0.8, int px = 520)
        : o_(origin), s_(px / (2.0 * half_span_m)), px_(px) {
        trail_ = cv::Mat(px, px, CV_8UC3, cv::Scalar(26, 26, 28));
        for (double g = -half_span_m; g <= half_span_m + 1e-9; g += 0.2) {
            cv::line(trail_, toPix(cv::Vec2d(o_[0] + g, o_[1] - half_span_m)),
                     toPix(cv::Vec2d(o_[0] + g, o_[1] + half_span_m)), cv::Scalar(58, 58, 62), 1);
            cv::line(trail_, toPix(cv::Vec2d(o_[0] - half_span_m, o_[1] + g)),
                     toPix(cv::Vec2d(o_[0] + half_span_m, o_[1] + g)), cv::Scalar(58, 58, 62), 1);
        }
    }

    void addTrail(const cv::Vec2d& c) {
        cv::circle(trail_, toPix(c), 1, cv::Scalar(90, 180, 240), -1);
    }

    cv::Mat render(const Ekf& ekf, bool has_obs, const cv::Vec3d& obs, int obs_board) const {
        cv::Mat img = trail_.clone();
        const cv::Vec2d c(ekf.get(Ekf::kCx), ekf.get(Ekf::kCy));
        cv::circle(img, toPix(c), cvRound(ekf.get(Ekf::kR) * s_), cv::Scalar(95, 95, 100), 1, cv::LINE_AA);

        for (int b = 0; b < armor_geom::kCount; ++b) {
            const cv::Vec3d p = ekf.armorCenter(b);
            const cv::Point q = toPix(cv::Vec2d(p[0], p[1]));
            const cv::Scalar col = (b == obs_board) ? cv::Scalar(60, 220, 255)
                                                    : cv::Scalar(200, 160, 60);
            cv::drawMarker(img, q, col, cv::MARKER_SQUARE, 9, (b == obs_board) ? 2 : 1);
            cv::putText(img, std::to_string(b + 1), q + cv::Point(7, -7),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv::LINE_AA);
        }
        if (has_obs) {
            cv::drawMarker(img, toPix(cv::Vec2d(obs[0], obs[1])), cv::Scalar(80, 240, 80),
                           cv::MARKER_CROSS, 13, 2);
        }
        // 车中心 3σ 误差椭圆：对 P 的 x-y 子块特征分解，半轴 = 3√λ
        {
            cv::Mat sub = (cv::Mat_<double>(2, 2) << ekf.cov(Ekf::kCx, Ekf::kCx),
                                                     ekf.cov(Ekf::kCx, Ekf::kCy),
                                                     ekf.cov(Ekf::kCy, Ekf::kCx),
                                                     ekf.cov(Ekf::kCy, Ekf::kCy));
            cv::Mat vals, vecs;
            if (cv::eigen(sub, vals, vecs) && vals.at<double>(0) > 0.0) {
                const double a = 3.0 * std::sqrt(vals.at<double>(0)) * s_;
                const double b = 3.0 * std::sqrt(std::max(vals.at<double>(1), 0.0)) * s_;
                // 图上 y 轴朝下，特征向量的 y 分量要取反才是屏幕角度
                const double ang =
                    std::atan2(-vecs.at<double>(0, 1), vecs.at<double>(0, 0)) * 180.0 / CV_PI;
                cv::ellipse(img, toPix(c), cv::Size(cvRound(a), cvRound(b)), ang, 0.0, 360.0,
                            cv::Scalar(120, 120, 235), 1, cv::LINE_AA);
            }
        }
        cv::circle(img, toPix(c), 4, cv::Scalar(70, 70, 240), -1);

        cv::putText(img, "top view (world x-y)   grid 0.2 m", cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(190, 190, 200), 1, cv::LINE_AA);
        cv::putText(img, "red=center  box=EKF board  cross=observation", cv::Point(10, 38),
                    cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(150, 150, 160), 1, cv::LINE_AA);
        cv::putText(img, "ellipse = 3 sigma of center", cv::Point(10, 56),
                    cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(120, 120, 200), 1, cv::LINE_AA);
        return img;
    }

private:
    cv::Point toPix(const cv::Vec2d& p) const {
        return cv::Point(cvRound(px_ * 0.5 + (p[0] - o_[0]) * s_),
                         cvRound(px_ * 0.5 - (p[1] - o_[1]) * s_));
    }
    cv::Vec2d o_;
    double s_;
    int px_;
    cv::Mat trail_;
};

// 右上角状态面板。putText 画不了中文，全用字母符号；
// 每个量后面的 +- 是协方差给出的 1σ
void drawPanel(cv::Mat& frame, const Ekf& ekf, bool inited, int boot_done,
               int obs_board, double resid) {
    const int line_h = 26;
    const int margin = 12;
    const int w = 420;
    const int h = line_h * 9 + margin;

    cv::Rect box(frame.cols - w - margin, margin, w, h);
    cv::Mat roi = frame(box);
    cv::Mat shade(roi.size(), roi.type(), cv::Scalar(20, 20, 20));
    cv::addWeighted(roi, 0.25, shade, 0.75, 0.0, roi);

    auto put = [&](int row, const std::string& s, cv::Scalar color) {
        cv::putText(frame, s, cv::Point(box.x + 10, box.y + 20 + row * line_h),
                    cv::FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv::LINE_AA);
    };

    const cv::Scalar white(255, 255, 255);
    const cv::Scalar cyan(255, 220, 90);
    const cv::Scalar warm(120, 200, 255);

    if (!inited) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "EKF init  %d / %d frames", boot_done, kBootstrapFrames);
        put(0, "EKF: initialising", white);
        put(1, buf, warm);
        return;
    }

    const cv::Scalar dim(150, 150, 160);

    char buf[160];
    std::snprintf(buf, sizeof(buf), "c  ( %6.3f, %6.3f ) m", ekf.get(Ekf::kCx), ekf.get(Ekf::kCy));
    put(0, buf, cyan);
    std::snprintf(buf, sizeof(buf), "   +- ( %5.3f, %5.3f )", ekf.sigma(Ekf::kCx), ekf.sigma(Ekf::kCy));
    put(1, buf, dim);
    std::snprintf(buf, sizeof(buf), "v  ( %6.3f, %6.3f ) m/s", ekf.get(Ekf::kVx), ekf.get(Ekf::kVy));
    put(2, buf, cyan);
    std::snprintf(buf, sizeof(buf), "   +- ( %5.3f, %5.3f )", ekf.sigma(Ekf::kVx), ekf.sigma(Ekf::kVy));
    put(3, buf, dim);
    std::snprintf(buf, sizeof(buf), "za %6.3f +-%.3f m  vz %6.3f +-%.3f", ekf.get(Ekf::kZa),
                  ekf.sigma(Ekf::kZa), ekf.get(Ekf::kVz), ekf.sigma(Ekf::kVz));
    put(4, buf, white);
    std::snprintf(buf, sizeof(buf), "th %6.2f +-%.2f deg  w %5.2f +-%.2f",
                  ekf.get(Ekf::kTheta) * 180.0 / CV_PI, ekf.sigma(Ekf::kTheta) * 180.0 / CV_PI,
                  ekf.get(Ekf::kOmega), ekf.sigma(Ekf::kOmega));
    put(5, buf, white);
    std::snprintf(buf, sizeof(buf), "r   %6.3f +-%.4f m", ekf.get(Ekf::kR), ekf.sigma(Ekf::kR));
    put(6, buf, white);
    if (obs_board >= 0) {
        std::snprintf(buf, sizeof(buf), "obs armor_%d  resid %.3f m  NIS %.1f", obs_board + 1,
                      resid, ekf.lastNis());
    } else {
        std::snprintf(buf, sizeof(buf), "obs none (predict only)");
    }
    put(7, buf, cv::Scalar(140, 255, 140));
    put(8, "+- = 1 sigma from covariance", cv::Scalar(170, 170, 180));
}

} // namespace

int runEkf(const std::string& video_path, const std::string& imu_path,
           const std::string& config_path, const std::string& method_arg,
           const std::string& model_dir, const std::string& video_out,
           const std::string& csv_out) {
    std::vector<std::string> methods;
    if (!parseMethods(method_arg, methods)) return -1;
    if (methods.size() > 1) methods.resize(1);
    const std::string method = methods[0];

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

    const std::vector<ExtrinsicOption> opts = enumerateExtrinsics();
    if (kExtrinsicIndex < 0 || kExtrinsicIndex >= static_cast<int>(opts.size())) {
        std::cerr << "[错误] 外参编号越界\n";
        return -1;
    }
    const ExtrinsicOption& extr = opts[kExtrinsicIndex];

    std::cout << "========== ekf 模式 ==========\n"
              << "视频: " << video_path << "\n"
              << "IMU : " << imu_path << "  " << imu.size() << " 行"
              << "（采集时刻已退 " << kImuOffsetSec * 1000 << " ms）\n"
              << "内参: " << config_path << "\n"
              << "方案: " << method << "\n"
              << "外参: #" << kExtrinsicIndex << "  " << extr.name << "\n";

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

    // ---------- 第一遍：逐帧识别 + PnP，只存位姿，不留图像 ----------
    std::vector<FramePoses> frames;
    cv::Mat frame;
    int frame_index = 0;
    long long solved_total = 0;
    const auto t_begin = std::chrono::steady_clock::now();

    while (cap.read(frame) && !frame.empty()) {
        ++frame_index;
        FramePoses rec;
        rec.frame = frame_index;
        if (frame_index - 1 < static_cast<int>(imu.size())) {
            rec.capture_ts = imu[frame_index - 1].capture_time;
        }
        for (const auto& d : detectors.detect(frame, method)) {
            const PoseResult p = solver.solve(d);
            if (!p.success) continue;
            ArmorPoseCam a;
            a.frame = frame_index;
            a.digit = d.digit;
            a.pos = p.tvec / 1000.0;
            a.rvec = p.rvec;
            a.reproj = p.reproj_error;
            a.image_points = d.image_points;
            rec.armors.push_back(a);
        }
        solved_total += static_cast<long long>(rec.armors.size());
        frames.push_back(rec);

        if (frame_index % 120 == 0) {
            std::cout << "  [识别] " << frame_index << " 帧" << std::endl;
        }
    }
    cap.release();
    const double t_detect = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_begin).count();
    std::cout << "识别完成: " << frames.size() << " 帧，解算 " << solved_total
              << " 块，耗时 " << std::fixed << std::setprecision(1) << t_detect << " s\n";

    if (solved_total < 30) {
        std::cerr << "[错误] 解算成功的板太少，没法跑 EKF\n";
        return -1;
    }

    // ---------- 启动阶段：用开头几帧的板位拟合圆，拿圆心/半径/高度当初值 ----------
    std::vector<cv::Vec2d> boot_pts;
    std::vector<double> boot_heights;
    const int boot_n = std::min<int>(kBootstrapFrames, static_cast<int>(frames.size()));
    for (int k = 0; k < boot_n; ++k) {
        if (frames[k].frame - 1 >= static_cast<int>(imu.size())) break;
        const cv::Mat R_wc = worldRotation(imu[frames[k].frame - 1].quat, extr);
        for (const auto& a : frames[k].armors) {
            const cv::Vec3d w = camToWorld(a.pos, R_wc);
            boot_pts.emplace_back(w[0], w[1]);
            boot_heights.push_back(w[2]);
        }
    }

    cv::Vec2d boot_center(0, 0);
    double boot_radius = armor_geom::kRadiusMeanM;
    if (!fitCircle(boot_pts, boot_center, boot_radius)) {
        std::cerr << "[错误] 启动阶段的点太少，拟合不出圆心\n";
        return -1;
    }
    const double boot_height = meanOf(boot_heights);
    std::cout << std::setprecision(3) << "\n---- 启动阶段（前 " << boot_n << " 帧，" << boot_pts.size() << " 个板位）----\n"
              << "拟合圆心 = (" << boot_center[0] << ", " << boot_center[1] << ") m\n"
              << "拟合半径 = " << boot_radius << " m"
              << "（官方 /tf 真值 " << armor_geom::kRadiusMeanM << " m）\n"
              << "平均板高 = " << boot_height << " m\n";

    // ---------- 第二遍：跑 EKF，画重投影，写视频和 CSV ----------
    cv::VideoCapture cap2;
    double fps2 = fps;
    int total2 = 0;
    if (!camera::openVideo(video_path, cap2, &fps2, &total2)) return -1;

    cv::VideoWriter writer;
    bool video_ok = false;
    if (!video_out.empty()) {
        writer.open(video_out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps2,
                    cv::Size(static_cast<int>(cap2.get(cv::CAP_PROP_FRAME_WIDTH)),
                             static_cast<int>(cap2.get(cv::CAP_PROP_FRAME_HEIGHT))));
        video_ok = writer.isOpened();
        if (!video_ok) {
            std::cerr << "[警告] 视频写不进去: " << video_out << "，只输出 CSV\n";
        }
    }

    std::ofstream csv(csv_out);
    if (!csv) {
        std::cerr << "[错误] 写不了 CSV: " << csv_out << "\n";
        return -1;
    }
    // 末尾的 s_* 是协方差给出的 1 倍标准差，nis 是归一化新息平方（3 维观测理论值 3）。
    // 放在末尾而不是插在中间，是为了不让老的状态列号发生位移
    csv << "frame,t,cx,vx,cy,vy,za,vz,theta,omega,r,obs_board,resid,inited,"
           "s_cx,s_vx,s_cy,s_vy,s_za,s_vz,s_theta,s_omega,s_r,nis\n";
    csv << std::fixed << std::setprecision(6);

    Ekf ekf;
    bool ekf_ready = false;
    double theta0 = 0.0;
    int boot_seen = 0;

    const double kArmorW = PoseSolver::kDefaultArmorWidthMm / 1000.0;
    const double kArmorH = PoseSolver::kDefaultArmorHeightMm / 1000.0;

    int idx = 0;
    int updates = 0, jumps = 0, misses = 0;
    double nis_sum = 0.0;   // 新息检验用，判断协方差给得准不准
    int nis_n = 0;

    // RV_NOWIN=1 完全不碰窗口系统。WSLg 的图形栈坏掉时窗口会卡在 waitKey 里关不掉，
    // 这个开关用来绕开它，改由下面的落盘路径输出同样的画面
    const bool no_window = (std::getenv("RV_NOWIN") != nullptr);
    bool show_win = !no_window;   // 没有图形界面时会自动关掉，只写视频
    bool paused = false;
    bool user_quit = false;
    TopView top(boot_center);

    // 显式建窗并给定尺寸。默认的 WINDOW_AUTOSIZE 在 WSLg 的 GTK 后端下，首次 imshow
    // 经常只建出窗口而不绘制内容，表现为空白框；WINDOW_NORMAL + resizeWindow 可以正常刷新
    const std::string frame_dir = "results/frames";
    const bool save_frames = true;      // 关掉窗口也照样出图，方便在 VS Code 里直接看
    constexpr int kSaveEveryN = 30;     // 每 30 帧存一张，531 帧约 18 组
    if (save_frames) {
        std::error_code ec;
        std::filesystem::create_directories(frame_dir, ec);
        if (ec) std::cerr << "[警告] 建不了 " << frame_dir << "：" << ec.message() << "\n";
    }

    if (show_win) {
        try {
            cv::namedWindow("ekf", cv::WINDOW_NORMAL);
            cv::resizeWindow("ekf", 960, 720);
            cv::namedWindow("top view", cv::WINDOW_NORMAL);
            cv::resizeWindow("top view", 560, 560);
            cv::waitKey(1); // 让窗口管理器先把窗口映射出来
        } catch (const cv::Exception&) {
            std::cerr << "[提示] 窗口建不起来（没有图形界面），只写视频\n";
            show_win = false;
        }
    }

    while (cap2.read(frame) && !frame.empty()) {
        if (idx >= static_cast<int>(frames.size())) break;
        const FramePoses& rec = frames[idx];
        ++idx;

        const cv::Mat R_wc = (rec.frame - 1 < static_cast<int>(imu.size()))
                                 ? worldRotation(imu[rec.frame - 1].quat, extr)
                                 : cv::Mat::eye(3, 3, CV_64F);

        // 启动阶段：攒够帧数就用拟合结果初始化，theta 由首帧观测直接反解
        if (!ekf_ready) {
            ++boot_seen;
            if (boot_seen >= boot_n && !rec.armors.empty()) {
                const cv::Vec3d w = camToWorld(rec.armors.front().pos, R_wc);
                const double base = std::atan2(w[1] - boot_center[1], w[0] - boot_center[0]);
                // digit 能认出来就用它当板号（数字直接对应板号），认不出先按 1 号板算，
                // 后续匹配会按几何关系自己收敛，theta 与板号是同一个自由度
                const int board = (rec.armors.front().digit >= 1 && rec.armors.front().digit <= 4)
                                      ? rec.armors.front().digit - 1
                                      : 0;
                theta0 = base - armor_geom::kPhiDeg[board] * CV_PI / 180.0;
                while (theta0 >= CV_PI) theta0 -= 2 * CV_PI;
                while (theta0 < -CV_PI) theta0 += 2 * CV_PI;
                ekf.init(rec.capture_ts, boot_center, boot_radius, theta0, boot_height);
                ekf_ready = true;
                std::cout << "EKF 初始化于第 " << rec.frame << " 帧（t=" << rec.capture_ts
                          << " s），theta0 = " << theta0 * 180.0 / CV_PI << " deg，板号 "
                          << board + 1 << "\n\n";
            }
        } else {
            ekf.predict(rec.capture_ts);
        }

        int obs_board = -1;
        double resid = -1.0;
        bool obs_valid = false;
        cv::Vec3d obs_p(0, 0, 0);

        if (ekf_ready) {
            for (const auto& a : rec.armors) {
                const cv::Vec3d p_w = camToWorld(a.pos, R_wc);
                int b = ekf.matchByAngle(p_w, kMatchAngleRad);
                if (b < 0) {
                    // 角度差太大：可能刚换成另一块板，先用观测重定 theta 再配一次
                    ekf.resetTheta(p_w);
                    b = ekf.matchByAngle(p_w, kReacquireAngleRad);
                    if (b < 0) {
                        ++misses;
                        continue; // 重定之后还是配不上，这帧观测丢掉
                    }
                    ++jumps;
                }
                resid = cv::norm(ekf.armorCenter(b) - p_w);
                ekf.update(b, p_w);
                nis_sum += ekf.lastNis();
                ++nis_n;
                obs_board = b;
                obs_valid = true;
                obs_p = p_w;
                ++updates;
            }
        }

        // ---- 画图 ----
        // 实际检出的板：绿框（用原始角点，和识别结果对得上）
        for (const auto& a : rec.armors) {
            for (int k = 0; k < 4; ++k) {
                cv::line(frame, a.image_points[k], a.image_points[(k + 1) % 4],
                         cv::Scalar(80, 230, 80), 2, cv::LINE_AA);
            }
            if (a.digit >= 0) {
                cv::putText(frame, std::to_string(a.digit),
                            a.image_points[0] + cv::Point2f(0, -6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(80, 230, 80), 2, cv::LINE_AA);
            }
        }

        // 四块板的预测位置重投影：青色细框
        if (ekf_ready) {
            for (int b = 0; b < armor_geom::kCount; ++b) {
                std::array<cv::Point3d, 4> corners;
                ekf.armorCorners(b, kArmorW, kArmorH, corners);
                std::vector<cv::Point3d> obj(corners.begin(), corners.end());
                const std::vector<cv::Point2d> img =
                    projectWorld(obj, R_wc, intr.camera_matrix, intr.dist_coeffs);

                bool visible = true;
                for (const auto& p : img) {
                    if (!std::isfinite(p.x) || p.x < -frame.cols || p.x > 2 * frame.cols ||
                        p.y < -frame.rows || p.y > 2 * frame.rows) {
                        visible = false;
                        break;
                    }
                }
                if (!visible) continue;

                const cv::Scalar color = (b == obs_board) ? cv::Scalar(60, 200, 255)
                                                          : cv::Scalar(200, 160, 60);
                for (int k = 0; k < 4; ++k) {
                    cv::line(frame, toInt(img[k]), toInt(img[(k + 1) % 4]), color,
                             (b == obs_board) ? 2 : 1, cv::LINE_AA);
                }
                cv::putText(frame, "E" + std::to_string(b + 1),
                            toInt(img[3]) + cv::Point(-4, 16),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
            }
        }

        drawPanel(frame, ekf, ekf_ready, boot_seen, obs_board, resid);

        // ---- 实时显示 ----
        static bool announced = false;
        if (show_win) {
            try {
                cv::Mat disp = frame;
                if (disp.cols > 1280) {
                    cv::resize(disp, disp,
                               cv::Size(1280, cvRound(disp.rows * 1280.0 / disp.cols)));
                }
                cv::imshow("ekf", disp.isContinuous() ? disp : disp.clone());
                if (ekf_ready) {
                    const cv::Mat view = top.render(ekf, obs_valid, obs_p, obs_board);
                    cv::imshow("top view", view.isContinuous() ? view : view.clone());
                }
                if (!announced) {
                    announced = true;
                    std::cout << "[显示] 已打开 ekf / top view 两个窗口"
                                 "（空格暂停，q 或 ESC 结束）\n";
                }
                const int key = cv::waitKey(paused ? 0 : 1);
                if (key == 'q' || key == 27) {
                    user_quit = true;
                } else if (key == ' ') {
                    paused = !paused; // 空格暂停，方便逐帧看
                }
            } catch (const cv::Exception&) {
                std::cerr << "[提示] 窗口开不起来（可能没有图形界面），只写视频\n";
                show_win = false;
            }
        }

        if (ekf_ready) {
            static const int kStateOrder[Ekf::kDim] = {Ekf::kCx, Ekf::kVx, Ekf::kCy, Ekf::kVy,
                                                       Ekf::kZa, Ekf::kVz, Ekf::kTheta,
                                                       Ekf::kOmega, Ekf::kR};
            csv << rec.frame << "," << rec.capture_ts
                << "," << ekf.get(Ekf::kCx) << "," << ekf.get(Ekf::kVx)
                << "," << ekf.get(Ekf::kCy) << "," << ekf.get(Ekf::kVy)
                << "," << ekf.get(Ekf::kZa) << "," << ekf.get(Ekf::kVz)
                << "," << ekf.get(Ekf::kTheta) << "," << ekf.get(Ekf::kOmega)
                << "," << ekf.get(Ekf::kR)
                << "," << obs_board + 1 << "," << resid << ",1";
            for (int k = 0; k < Ekf::kDim; ++k) csv << "," << ekf.sigma(kStateOrder[k]);
            csv << "," << ekf.lastNis() << "\n";
        }

        if (writer.isOpened()) writer.write(frame);

        // 落盘路径：没有窗口时靠这些图看结果，内容和窗口里画的一致
        if (save_frames && idx % kSaveEveryN == 0) {
            char fp[256];
            std::snprintf(fp, sizeof(fp), "%s/frame_%04d.jpg", frame_dir.c_str(), idx);
            cv::imwrite(fp, frame);
            if (ekf_ready) {
                std::snprintf(fp, sizeof(fp), "%s/topview_%04d.png", frame_dir.c_str(), idx);
                cv::imwrite(fp, top.render(ekf, obs_valid, obs_p, obs_board));
            }
        }

        if (ekf_ready) top.addTrail(cv::Vec2d(ekf.get(Ekf::kCx), ekf.get(Ekf::kCy)));
        if (user_quit) {
            std::cout << "  [提示] 用户提前结束\n";
            break;
        }

        if (idx % 120 == 0) {
            std::cout << "  [EKF] " << idx << " 帧" << std::endl;
        }
    }
    cap2.release();
    if (writer.isOpened()) writer.release();
    if (save_frames) {
        if (ekf_ready) {
            cv::Mat last = top.render(ekf, false, cv::Vec3d(0, 0, 0), -1);
            cv::imwrite(frame_dir + "/topview_final.png", last);
        }
        std::cout << "关键帧与俯视图已存到 " << frame_dir << "/\n";
    }

    if (show_win) {
        std::cout << "\n处理结束，窗口保留中——按任意键关闭\n";
        cv::waitKey(0);
    }
    cv::destroyAllWindows();
    csv.close();

    std::cout << "\n---- EKF 统计 ----\n"
              << "观测更新 " << updates << " 次，跳变重定 " << jumps
              << " 次，丢弃 " << misses << " 次\n"
              << "最终状态：c=(" << ekf.get(Ekf::kCx) << ", " << ekf.get(Ekf::kCy) << ") m"
              << "  za=" << ekf.get(Ekf::kZa) << " m"
              << "  theta=" << ekf.get(Ekf::kTheta) * 180.0 / CV_PI << " deg"
              << "  r=" << ekf.get(Ekf::kR) << " m\n"
              << "协方差 1sigma：c=(" << ekf.sigma(Ekf::kCx) << ", " << ekf.sigma(Ekf::kCy) << ") m"
              << "  v=(" << ekf.sigma(Ekf::kVx) << ", " << ekf.sigma(Ekf::kVy) << ") m/s"
              << "  za=" << ekf.sigma(Ekf::kZa) << " m\n"
              << "               theta=" << ekf.sigma(Ekf::kTheta) * 180.0 / CV_PI << " deg"
              << "  omega=" << ekf.sigma(Ekf::kOmega) << " rad/s"
              << "  r=" << ekf.sigma(Ekf::kR) << " m\n";
    if (nis_n > 0) {
        std::cout << "新息检验 NIS 均值 " << nis_sum / nis_n
                  << "（3 维观测的理论值是 3.0，明显偏大说明协方差被低估）\n";
    }
    std::cout << "状态序列: " << csv_out << "\n";
    if (show_win)
        std::cout << "（窗口已弹出：视频画面与俯视轨迹，空格暂停、q 或 ESC 结束）\n";
    else
        std::cout << "（RV_NOWIN 已设，不弹窗口；关键帧与俯视图见 " << frame_dir << "/）\n";
    if (!video_ok) {
        std::cout << "[提示] 视频没写出来，检查 " << video_out << " 所在目录是否存在\n";
    } else {
        std::cout << "可视化视频: " << video_out << "\n";
    }
    std::cout << "===============================\n";
    return 0;
}
