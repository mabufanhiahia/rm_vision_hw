#include "pose_solver.h"

#include <cmath>

PoseSolver::PoseSolver(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                       float armor_width, float armor_height)
    : camera_matrix_(camera_matrix), dist_coeffs_(dist_coeffs) {
    // 物体坐标系原点在装甲板中心，x 右、y 下、z 垂直板面朝相机；
    // 顺序必须与 ArmorDetection::image_points 一致。图像 y 向下，所以"上"对应的 y 为负。
    const float w = armor_width / 2.0f;
    const float h = armor_height / 2.0f;
    object_points_ = {
        cv::Point3f(-w, -h, 0), // 左上
        cv::Point3f( w, -h, 0), // 右上
        cv::Point3f( w,  h, 0), // 右下
        cv::Point3f(-w,  h, 0)  // 左下
    };
}

PoseResult PoseSolver::solve(const ArmorDetection& detection) {
    PoseResult result;

    const std::vector<cv::Point2f> img_pts(detection.image_points.begin(),
                                           detection.image_points.end());

    // 4 个共面点有两个镜像解，重投影几乎一样。solvePnP 只返回其中一个，可能拿到"背面"
    // 那个（image1 就撞过：x 解成 −161.6 而真值 +150.8）。所以用 solvePnPGeneric
    // 取全部解，再按物理条件挑：板面法向朝相机（R(2,2)<0）、且板在相机前方（tz>0）。
    std::vector<cv::Mat> rvecs, tvecs;
    const int nsol = cv::solvePnPGeneric(object_points_, img_pts, camera_matrix_, dist_coeffs_,
                                         rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
    if (nsol > 0) {
        int pick = 0;
        for (int k = 0; k < nsol; ++k) {
            cv::Mat R;
            cv::Rodrigues(rvecs[k], R);
            const double nz = R.at<double>(2, 2);
            const double tz = tvecs[k].at<double>(2);
            if (tz > 0.0 && nz < 0.0) { pick = k; break; }
        }
        result.rvec = cv::Vec3d(rvecs[pick].at<double>(0), rvecs[pick].at<double>(1),
                                rvecs[pick].at<double>(2));
        result.tvec = cv::Vec3d(tvecs[pick].at<double>(0), tvecs[pick].at<double>(1),
                                tvecs[pick].at<double>(2));
    } else {
        // 四点几乎共线时 IPPE 退化，退回迭代解法
        const bool ok = cv::solvePnP(object_points_, img_pts, camera_matrix_, dist_coeffs_,
                                     result.rvec, result.tvec, false, cv::SOLVEPNP_ITERATIVE);
        if (!ok) {
            return result; // success=false、reproj_error=-1，表示 PnP 无解
        }
    }

    // 重投影误差只用来剔除坏解，不能拿来判断精度：4 点平面 PnP 恰好定解，
    // 误差常在 0.0x px，人为构造的完美矩形更是必然拟合得好。
    std::vector<cv::Point2f> reproj;
    cv::projectPoints(object_points_, result.rvec, result.tvec,
                      camera_matrix_, dist_coeffs_, reproj);
    double sum = 0.0;
    for (size_t i = 0; i < reproj.size(); ++i) {
        const double ex = static_cast<double>(reproj[i].x) - img_pts[i].x;
        const double ey = static_cast<double>(reproj[i].y) - img_pts[i].y;
        sum += std::sqrt(ex * ex + ey * ey);
    }
    result.reproj_error = sum / static_cast<double>(reproj.size());

    result.success = (result.reproj_error <= kMaxReprojError);
    return result;
}
