#include "ekf.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr double kTwoPi = 2.0 * CV_PI;

// 角度折回 [-pi, pi)：theta 累加 omega·dt 会一直长下去，不折回来的话
// 曲线图画到最后会是一条斜线，看不出来回摆动
double wrapAngle(double a) {
    while (a >= CV_PI) a -= kTwoPi;
    while (a < -CV_PI) a += kTwoPi;
    return a;
}

double phiOf(int board) {
    return armor_geom::kPhiDeg[board] * CV_PI / 180.0;
}

// 各板相对"四块平均"的偏置
double radiusOffset(int board) {
    return armor_geom::kRadiusM[board] - armor_geom::kRadiusMeanM;
}

double heightOffset(int board) {
    double mean = 0.0;
    for (int i = 0; i < armor_geom::kCount; ++i) mean += armor_geom::kHeightM[i];
    mean /= armor_geom::kCount;
    return armor_geom::kHeightM[board] - mean;
}

// 让 P 保持对称。标准 EKF 更新式 (I-KH)P 在数值上会慢慢失去对称性，
// 不对称越积越多后 S 求逆会抖，最后滤波器发散
void symmetrize(cv::Mat& P) {
    P = 0.5 * (P + P.t());
}

} // namespace

void Ekf::init(double t, const cv::Vec2d& center, double radius, double theta, double height) {
    last_t_ = t;
    inited_ = true;

    x_ = cv::Mat::zeros(kDim, 1, CV_64F);
    x_.at<double>(kCx) = center[0];
    x_.at<double>(kCy) = center[1];
    x_.at<double>(kZa) = height;
    x_.at<double>(kTheta) = wrapAngle(theta);
    x_.at<double>(kR) = radius;

    // 协方差初值。速度和角速度给得比位置宽，让前几帧观测说了算；
    // 半径按官方几何冻结（1e-4 m 相当于知道到 0.1mm），避免它去吸收水平方向的观测误差。
    P_ = cv::Mat::zeros(kDim, kDim, CV_64F);
    P_.at<double>(kCx, kCx) = 0.05 * 0.05;
    P_.at<double>(kCy, kCy) = 0.05 * 0.05;
    P_.at<double>(kVx, kVx) = 0.10 * 0.10;
    P_.at<double>(kVy, kVy) = 0.10 * 0.10;
    P_.at<double>(kZa, kZa) = 0.02 * 0.02;
    P_.at<double>(kVz, kVz) = 0.08 * 0.08;
    P_.at<double>(kTheta, kTheta) = 0.35 * 0.35;
    P_.at<double>(kOmega, kOmega) = 0.15 * 0.15;
    P_.at<double>(kR, kR) = 1e-4 * 1e-4;
}

void Ekf::predict(double t) {
    if (!inited_) return;
    double dt = t - last_t_;
    if (dt <= 1e-6) return;
    // 时间戳偶尔会抖，超过 0.25s 按 0.25s 算，免得一次大跳把状态拉飞
    if (dt > 0.25) dt = 0.25;
    last_t_ = t;

    x_.at<double>(kCx) += x_.at<double>(kVx) * dt;
    x_.at<double>(kCy) += x_.at<double>(kVy) * dt;
    x_.at<double>(kZa) += x_.at<double>(kVz) * dt;
    x_.at<double>(kTheta) = wrapAngle(x_.at<double>(kTheta) + x_.at<double>(kOmega) * dt);

    cv::Mat F = cv::Mat::eye(kDim, kDim, CV_64F);
    F.at<double>(kCx, kVx) = dt;
    F.at<double>(kCy, kVy) = dt;
    F.at<double>(kZa, kVz) = dt;
    F.at<double>(kTheta, kOmega) = dt;

    // 过程噪声用连续白噪声加速度模型的离散形式，
    // 位置-速度交叉项不能省，省了会让位置的不确定度偏小、观测权重被高估
    cv::Mat Q = cv::Mat::zeros(kDim, kDim, CV_64F);
    auto addPair = [&](int ip, int iv, double q) {
        const double dt2 = dt * dt, dt3 = dt2 * dt;
        Q.at<double>(ip, ip) += q * dt3 / 3.0;
        Q.at<double>(ip, iv) += q * dt2 / 2.0;
        Q.at<double>(iv, ip) += q * dt2 / 2.0;
        Q.at<double>(iv, iv) += q * dt;
    };
    addPair(kCx, kVx, cfg_.q_accel_xy);
    addPair(kCy, kVy, cfg_.q_accel_xy);
    addPair(kZa, kVz, cfg_.q_accel_z);
    addPair(kTheta, kOmega, cfg_.q_alpha);
    Q.at<double>(kOmega, kOmega) += cfg_.q_omega * dt;
    Q.at<double>(kR, kR) += cfg_.q_radius * dt;

    P_ = F * P_ * F.t() + Q;
    symmetrize(P_);
}

int Ekf::matchByAngle(const cv::Vec3d& p, double max_angle_rad, double* angle_err) const {
    if (!inited_) return -1;

    const double dx = p[0] - x_.at<double>(kCx);
    const double dy = p[1] - x_.at<double>(kCy);
    // 观测点几乎落在车心上时方位角没有意义，直接判为配不上
    if (std::hypot(dx, dy) < 0.05) return -1;

    const double base = std::atan2(dy, dx);
    const double cur = x_.at<double>(kTheta);
    int best = -1;
    double best_d = 1e9;
    for (int b = 0; b < armor_geom::kCount; ++b) {
        const double th = wrapAngle(base - phiOf(b));
        const double d = std::fabs(wrapAngle(th - cur));
        if (d < best_d) {
            best_d = d;
            best = b;
        }
    }
    if (angle_err) *angle_err = best_d;
    return (best_d <= max_angle_rad) ? best : -1;
}

void Ekf::resetTheta(const cv::Vec3d& p) {
    if (!inited_) return;
    const double dx = p[0] - x_.at<double>(kCx);
    const double dy = p[1] - x_.at<double>(kCy);
    if (std::hypot(dx, dy) < 1e-6) return;

    // 观测点相对车中心的方位角就是 theta+phi_j，对四块板各反推一个 theta，
    // 取最接近当前估计的那个——这样跳变只修正角度，不会把车"瞬移"到别的板上去
    const double base = std::atan2(dy, dx);
    const double cur = x_.at<double>(kTheta);
    double best = cur, best_d = 1e9;
    for (int b = 0; b < armor_geom::kCount; ++b) {
        const double th = wrapAngle(base - phiOf(b));
        const double d = std::fabs(wrapAngle(th - cur));
        if (d < best_d) {
            best_d = d;
            best = th;
        }
    }
    x_.at<double>(kTheta) = best;

    // 角度是重新给的而不是算出来的，原来的角度方差不再可信，放开到 8°
    P_.at<double>(kTheta, kTheta) = std::max(P_.at<double>(kTheta, kTheta), 0.14 * 0.14);
}

void Ekf::update(int board, const cv::Vec3d& p) {
    if (!inited_ || board < 0 || board >= armor_geom::kCount) return;

    const double th = x_.at<double>(kTheta);
    const double ang = th + phiOf(board);
    const double r = x_.at<double>(kR) + radiusOffset(board);
    const double ca = std::cos(ang), sa = std::sin(ang);

    cv::Mat h(3, 1, CV_64F);
    h.at<double>(0) = x_.at<double>(kCx) + r * ca;
    h.at<double>(1) = x_.at<double>(kCy) + r * sa;
    h.at<double>(2) = x_.at<double>(kZa) + heightOffset(board);

    // H 在当前预测处求（课件 slide 15：F 在上一帧估计处，H 在本帧预测处）
    cv::Mat H = cv::Mat::zeros(3, kDim, CV_64F);
    H.at<double>(0, kCx) = 1.0;
    H.at<double>(0, kTheta) = -r * sa;
    H.at<double>(0, kR) = ca;
    H.at<double>(1, kCy) = 1.0;
    H.at<double>(1, kTheta) = r * ca;
    H.at<double>(1, kR) = sa;
    H.at<double>(2, kZa) = 1.0;

    cv::Mat R = cv::Mat::zeros(3, 3, CV_64F);
    R.at<double>(0, 0) = cfg_.sigma_xy * cfg_.sigma_xy;
    R.at<double>(1, 1) = cfg_.sigma_xy * cfg_.sigma_xy;
    R.at<double>(2, 2) = cfg_.sigma_z * cfg_.sigma_z;

    cv::Mat y(3, 1, CV_64F);
    y.at<double>(0) = p[0] - h.at<double>(0);
    y.at<double>(1) = p[1] - h.at<double>(1);
    y.at<double>(2) = p[2] - h.at<double>(2);

    cv::Mat S = H * P_ * H.t() + R;
    const cv::Mat Sinv = S.inv(cv::DECOMP_SVD);
    cv::Mat K = P_ * H.t() * Sinv;

    // 归一化新息平方，协方差是否可信就看它长期是否稳定在观测维数附近
    // y.t()*Sinv*y 是 cv::MatExpr（延迟求值），不能直接 .at()，先落成 cv::Mat 再取标量
    const cv::Mat nis_mat = y.t() * Sinv * y;
    last_nis_ = nis_mat.at<double>(0, 0);

    x_ += K * y;
    x_.at<double>(kTheta) = wrapAngle(x_.at<double>(kTheta));

    P_ = (cv::Mat::eye(kDim, kDim, CV_64F) - K * H) * P_;
    symmetrize(P_);
}

double Ekf::sigma(int i) const {
    if (!inited_ || i < 0 || i >= kDim) return 0.0;
    const double v = P_.at<double>(i, i);
    return v > 0.0 ? std::sqrt(v) : 0.0;
}

double Ekf::cov(int i, int j) const {
    if (!inited_ || i < 0 || i >= kDim || j < 0 || j >= kDim) return 0.0;
    return P_.at<double>(i, j);
}

cv::Vec3d Ekf::center() const {
    return cv::Vec3d(x_.at<double>(kCx), x_.at<double>(kCy), x_.at<double>(kZa));
}

double Ekf::speed() const {
    return std::hypot(x_.at<double>(kVx), x_.at<double>(kVy));
}

cv::Vec3d Ekf::armorCenter(int board) const {
    const double ang = x_.at<double>(kTheta) + phiOf(board);
    const double r = x_.at<double>(kR) + radiusOffset(board);
    return cv::Vec3d(x_.at<double>(kCx) + r * std::cos(ang),
                     x_.at<double>(kCy) + r * std::sin(ang),
                     x_.at<double>(kZa) + heightOffset(board));
}

void Ekf::armorCorners(int board, double width_m, double height_m,
                       std::array<cv::Point3d, 4>& out) const {
    const cv::Vec3d c = armorCenter(board);
    const double ang = x_.at<double>(kTheta) + phiOf(board);
    // 板法向朝车外，水平切向沿板宽；板竖直安装，高边沿 z
    const cv::Vec3d tangent(-std::sin(ang), std::cos(ang), 0.0);
    const cv::Vec3d up(0.0, 0.0, 1.0);

    const double hw = width_m * 0.5;
    const double hh = height_m * 0.5;
    out[0] = c - tangent * hw + up * hh;
    out[1] = c + tangent * hw + up * hh;
    out[2] = c + tangent * hw - up * hh;
    out[3] = c - tangent * hw - up * hh;
}
