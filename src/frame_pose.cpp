#include "frame_pose.h"

#include <algorithm>
#include <numeric>

namespace {

// 生成 3 个坐标轴的全排列
std::vector<std::array<int, 3>> permutations3() {
    std::vector<std::array<int, 3>> out;
    std::array<int, 3> p = {0, 1, 2};
    do {
        out.push_back(p);
    } while (std::next_permutation(p.begin(), p.end()));
    return out;
}

// 排列的奇偶性，用来保证生成的旋转是右手系（行列式 +1）
int permutationSign(const std::array<int, 3>& p) {
    int sign = 1;
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            if (p[i] > p[j]) sign = -sign;
        }
    }
    return sign;
}

std::string axisName(int i) {
    return (i == 0) ? "X" : ((i == 1) ? "Y" : "Z");
}

} // namespace

std::vector<ExtrinsicOption> enumerateExtrinsics() {
    std::vector<ExtrinsicOption> opts;

    for (const bool transpose : {false, true}) {
        for (const auto& p : permutations3()) {
            // R_ic 的第 i 行把 ±1 放在第 p[i] 列上：表示相机第 p[i] 轴对应 IMU 第 i 轴
            for (int mask = 0; mask < 8; ++mask) {
                const int sx = (mask & 1) ? -1 : 1;
                const int sy = (mask & 2) ? -1 : 1;
                const int sz = (mask & 4) ? -1 : 1;
                if (permutationSign(p) * sx * sy * sz != 1) continue; // 行列式必须为 +1

                cv::Mat R = cv::Mat::zeros(3, 3, CV_64F);
                R.at<double>(0, p[0]) = sx;
                R.at<double>(1, p[1]) = sy;
                R.at<double>(2, p[2]) = sz;

                ExtrinsicOption opt;
                opt.R_ic = R;
                opt.transpose_imu = transpose;
                opt.name = std::string(transpose ? "imu^T" : "imu") + " C:(" +
                           (sx < 0 ? "-" : "+") + axisName(p[0]) + "," +
                           (sy < 0 ? "-" : "+") + axisName(p[1]) + "," +
                           (sz < 0 ? "-" : "+") + axisName(p[2]) + ")";
                opts.push_back(opt);
            }
        }
    }
    return opts;
}

cv::Mat worldRotation(const cv::Vec4d& quat, const ExtrinsicOption& opt) {
    cv::Mat R_wi = ImuReader::quatToRot(quat);
    if (opt.transpose_imu) R_wi = R_wi.t();
    return R_wi * opt.R_ic; // 先转到 IMU 系，再按安装关系转到相机系
}

cv::Vec3d camToWorld(const cv::Vec3d& p_cam, const cv::Mat& R_wc) {
    const cv::Mat p(R_wc * cv::Mat(p_cam));
    return cv::Vec3d(p.at<double>(0), p.at<double>(1), p.at<double>(2));
}
