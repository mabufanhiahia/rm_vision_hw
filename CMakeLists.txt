# 必须最先声明：后面的命令和策略都依赖它
cmake_minimum_required(VERSION 3.16)
project(rm_vision_hw LANGUAGES CXX)

# ---- 全局编译设置 ----
set(CMAKE_CXX_STANDARD 17)             # 用到 std::filesystem 等 C++17 特性
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 生成 compile_commands.json，给 clangd 补全用
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)      # 默认 Release：带 -O 优化，跑视频才流畅
endif()

# ---- 依赖 1：OpenCV ----
# WSL/Ubuntu 里装好 libopencv-dev 后这行就能直接找到。
# 矩阵运算全部用 cv::Mat/cv::Vec，第三周的 EKF 也没有引入 Eigen。
find_package(OpenCV REQUIRED)

# ---- 依赖 2：ONNX Runtime（神经网络方案用）----
# 默认到 ~/onnxruntime-linux 找，也可以在配置时覆盖：
#   cmake -B build -DONNXRUNTIME_ROOT=/你的/onnxruntime路径
set(ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-linux" CACHE PATH "ONNX Runtime 解压根目录")
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    message(FATAL_ERROR
        "在 ${ONNXRUNTIME_ROOT}/lib 下找不到 libonnxruntime.so\n"
        "请确认 ONNX Runtime 已按 README 解压到位，或配置时指定：\n"
        "  cmake -B build -DONNXRUNTIME_ROOT=/你的/onnxruntime路径")
endif()

# ---- 目标：主程序 ----
# 显式列出源文件（比 file(GLOB) 规范：新增文件必须手动登记，避免"改了却没编译进去"）
# 按模块分组排列，和 include/ 下的头文件一一对应：
#   main / pipeline_*   入口与五条流水线（图片、视频、dump、ekf、plot）
#   detectors           两种识别方案的统一入口
#   camera / ground_truth  数据源与真值
#   traditional_detector / YOLO_detector / digit_classifier  识别与数字分类
#   pose_solver         位置解算
#   visualizer          结果展示
#   imu_reader / frame_pose  第三周：IMU 内录数据读取与相机系→世界系转换
#   ekf                 第三周：9 维扩展卡尔曼滤波
add_executable(rm_vision_hw
    src/main.cpp
    src/pipeline_image.cpp
    src/pipeline_video.cpp
    src/pipeline_dump.cpp
    src/pipeline_ekf.cpp
    src/pipeline_plot.cpp
    src/detectors.cpp
    src/camera.cpp
    src/ground_truth.cpp
    src/traditional_detector.cpp
    src/YOLO_detector.cpp
    src/digit_classifier.cpp
    src/pose_solver.cpp
    src/visualizer.cpp
    src/imu_reader.cpp
    src/frame_pose.cpp
    src/ekf.cpp
)

# 现代 CMake 写法：include/链接只作用于本 target，不污染全局
target_include_directories(rm_vision_hw PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_ROOT}/include
)
target_link_libraries(rm_vision_hw PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)

# 让程序运行时能自动找到 libonnxruntime.so
# （否则每次运行前都得手动 export LD_LIBRARY_PATH）
set_target_properties(rm_vision_hw PROPERTIES
    BUILD_RPATH   "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

# ---- OpenCV 组件要求 ----
# dnn：数字识别网络要用 cv::dnn::readNetFromONNX（见 src/digit_classifier.cpp）
# 如果 find_package 时没带上 dnn 组件，链接时会报 "undefined reference to cv::dnn::..."
if(OpenCV_VERSION VERSION_LESS "4.5.0")
    message(WARNING "OpenCV 版本较旧(${OpenCV_VERSION})，cv::dnn 加载 ONNX 可能出现算子不支持，建议升级到 4.5+")
endif()
