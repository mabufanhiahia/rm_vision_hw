#pragma once

#include <opencv2/opencv.hpp>
#include <array>
#include <string>

// 两种识别方案共同的输出结构，下游解算和画图只认它，不关心板是哪套算法找出来的。
struct ArmorDetection {
    // 四角像素坐标，顺序固定为 左上 → 右上 → 右下 → 左下，必须与 PoseSolver 的物点一一对应
    std::array<cv::Point2f, 4> image_points;

    int digit = -1;          // 装甲板上的数字 1~5，-1 表示没认出来
    float confidence = 0.0f; // 置信度 0~1，传统方案没有这个概念，保持 0
    std::string method;      // "traditional" 或 "nn"

    int class_id = -1;       // 仅神经网络方案填：兵种类别下标
    std::string class_name;  // 仅神经网络方案填：兵种类别名
};

struct PoseResult {
    // 相机系位置，单位毫米；x 右为正、y 下为正、z 沿光轴朝物体，参考点是装甲板中心
    cv::Vec3d tvec;
    cv::Vec3d rvec;              // 旋转向量，描述装甲板朝向
    bool success = false;        // 解算是否可信
    double reproj_error = -1.0;  // 重投影误差（像素），-1 表示还没算过
};
