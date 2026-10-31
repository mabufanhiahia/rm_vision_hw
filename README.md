# rm_vision_hw —— 27 赛季算法组视觉侧第二、三周作业

传统视觉 + 神经网络**双方案**的装甲板识别、PnP 位置解算与 EKF 整车状态估计
（C++ / OpenCV / ONNX Runtime）。

两套方案产出同一种数据结构 `ArmorDetection`，所以下游的位置解算、误差比对、
结果展示完全不需要知道这一帧是哪套算法检出来的。第三周在其上接了一条链路：
PnP 位姿按曝光时刻配 IMU 姿态转到世界系，送进 9 维 EKF 估计车中心、速度、朝向、角速度。

---

## 1. 快速开始

**必须在项目根目录 `rm_vision_hw/` 下执行**，代码里的默认路径都是相对它的。

```bash
# ---- 构建（配置只需首次或改了 CMakeLists 后执行一次）----
cmake -S . -B build
cmake --build build -j

# ---- 图片模式：检测 → PnP 解算 → 与真值比对 ----
./build/rm_vision_hw image                 # 处理 data/images/ 下全部图片（默认 + 传统方案）
./build/rm_vision_hw image data/images/image2.jpg   # 只处理这一张
./build/rm_vision_hw image data/images data/gt/ground_truth.csv   # 显式指定真值表
./build/rm_vision_hw image data/images/image1.jpg data/gt/ground_truth.csv \
                           config/camera_image.yaml both   # 两种方案都跑

# ---- 视频模式：逐帧检测 → 画框+数字 → 输出标注视频 ----
./build/rm_vision_hw video data/videos/armor_record.avi traditional
./build/rm_vision_hw video data/videos/armor_record.avi nn

# ---- 第三周：EKF 整车状态估计 ----
./build/rm_vision_hw dump    # 标定相机-IMU 外参，导出每帧位姿到 results/frame_poses.csv
./build/rm_vision_hw ekf     # 逐帧滤波，输出 results/ekf_states.csv 与 ekf_video.mp4
./build/rm_vision_hw plot    # 状态序列画成 results/ekf_curves.png

# ekf 模式会弹两个窗口（画面 + 俯视轨迹，空格暂停、q 退出）。WSLg 的图形栈偶尔会
# 让窗口白板甚至卡住关不掉，这时跳过窗口系统，改看落盘的图：
RV_NOWIN=1 ./build/rm_vision_hw ekf

# ---- 忘了参数？直接跑它会打印用法 ----
./build/rm_vision_hw
```

### 命令行参数

```
rm_vision_hw image [图片或图片目录] [真值csv] [内参yaml] [方案]
rm_vision_hw video [视频] [方案] [输出路径(可选)]
rm_vision_hw dump  [视频] [imu txt] [内参yaml] [方案] [输出csv] [外参编号(可选)]
rm_vision_hw ekf   [视频] [imu txt] [内参yaml] [方案] [输出视频] [输出csv]
rm_vision_hw plot  [状态csv] [输出png]
```

第二个参数给**目录**时会批量处理目录里所有 `.jpg/.png/.bmp`，每张图各打印一张误差表；
给**单个文件**就只处理那一张。

| 参数 | 默认值 |
|---|---|
| 图片 | `data/images`（目录下全部图片） |
| 真值csv | `data/gt/ground_truth.csv` |
| 内参yaml | `config/camera_image.yaml` |
| 视频 | `data/videos/armor_record.avi` |
| imu txt | `data/imu/imu_record.txt`（第三周） |
| 方案 | `traditional` |
| 第三周内参 | dump / ekf 模式默认 `config/camera_video.yaml` |

**方案取值**：`traditional`（传统视觉）/ `nn`（神经网络）/ `both`（两种都跑）。
图片模式下 `both` 会打印两张误差表；视频模式下 `both` 会输出两个视频文件。

### 输出

| 输出 | 内容 |
|---|---|
| 控制台 | 误差表：每块板的计算坐标 / 真值坐标 / Δx,Δy,Δz / 三维误差 / 重投影误差，末尾有统计 |
| 弹窗 | `imshow` 展示标注结果，按任意键或点窗口 × 关闭；视频模式下按 `q` / `ESC` 提前结束 |
| `results/detect_<图片名>_<方案>.jpg` | 图片标注图：绿框（传统）/ 青框（神经网络）= 检测结果，红色十字 = 真值投影位置 |
| `results/video_<方案>.avi` | 视频标注结果（画框 + 数字/兵种 + 左上角帧号与耗时） |
| `results/debug_mask.jpg` | 传统方案的蓝色优势分割掩膜，调阈值时看它最直观 |
| `results/frame_poses.csv` | 第三周 dump 模式：每帧每块板的相机系与世界系位姿 |
| `results/ekf_states.csv` | 第三周 ekf 模式：9 个状态量 + 9 个 σ + 每帧 NIS |
| `results/ekf_video.mp4` | 四块板重投影 + 右上角实时状态面板 |
| `results/ekf_curves.png` | 9 个状态量曲线，带 ±1σ 不确定度带 |
| `results/frames/` | 每 30 帧一张当前帧 `frame_XXXX.jpg` 与俯视图 `topview_XXXX.png`，不依赖窗口 |

---

## 2. 目录结构

```
rm_vision_hw/
├── CMakeLists.txt              # 构建脚本（显式列源文件，ONNX Runtime 路径可覆盖）
├── include/                    # 头文件（对外接口 + 注释即文档）
│   ├── types.h                 # 全项目共用的数据结构
│   ├── camera.h                # 数据源：内参 / 图片 / 视频读写
│   ├── ground_truth.h          # 真值表读取与三维最近邻匹配
│   ├── traditional_detector.h  # 【识别】传统视觉（蓝灯条 → 配对）
│   ├── YOLO_detector.h         # 【识别】神经网络（ONNX 装甲板检测）
│   ├── digit_classifier.h      # 【数字分类】小 MLP，识别 1~5 / outpost / guard / base
│   ├── detectors.h             # 两种方案的统一入口（含"结合数字网络配对"）
│   ├── pose_solver.h           # 【位置解算】PnP
│   ├── visualizer.h            # 【结果展示】画框 / 真值点 / 信息条
│   ├── imu_reader.h            # 【第三周】IMU 内录读取：时间戳 + 四元数
│   ├── frame_pose.h            # 【第三周】装甲板几何常量、外参候选、相机系→世界系
│   ├── ekf.h                   # 【第三周】9 维扩展卡尔曼滤波
│   └── pipeline.h              # 五条流水线：image / video / dump / ekf / plot
├── src/                        # 与上面一一对应的实现文件
├── config/
│   ├── camera_image.yaml       # 图片用内参（960×720，畸变全 0）
│   └── camera_video.yaml       # 视频用内参（1440×1080，带畸变）  两套不能混用
├── data/
│   ├── gt/ground_truth.csv     # 三张图的装甲板位置真值（相机系，米）
│   ├── images/                 # image1/2/3.jpg（960×720）
│   ├── imu/imu_record.txt      # 【第三周】IMU 内录，531 行（时间戳 w x y z）
│   └── videos/armor_record.avi
├── models/                     # ONNX 模型（见 models/README.md 的下载命令）
└── results/                    # 运行输出（gitignore，不入库）
```

模块按职责划分：**识别**、**数字分类**、**位置解算**、**结果展示**各自独立成头/源对，
第三周的 IMU 读取、坐标系转换、滤波、绘图同样各自成对，只依赖解算输出、不反向依赖。
`main.cpp` 只负责解析命令行和分发。

---

## 3. 环境依赖

- WSL Ubuntu / Linux，GCC ≥ 9（C++17）
- CMake ≥ 3.16
- OpenCV 4：`sudo apt install libopencv-dev`
  （用到 `core / imgproc / calib3d / highgui / videoio / dnn` 这几个模块，apt 包装一次全有）
- ONNX Runtime 1.17.1：已解压在 `~/onnxruntime-linux`（需要其中的 `include/` 和 `lib/`）

如果要换 ONNX Runtime 的路径：

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=/你的/onnxruntime路径
```

> 为什么数字识别不用 ONNX Runtime：那个 MLP 只有几层全连接，OpenCV 自带的
> `cv::dnn` 就够，能少一堆环境配置。装甲板检测网络算子多，才需要用 ORT。
> 详见 `include/digit_classifier.h` 的注释。

---

## 4. 重要：两套内参不要混用

| 数据 | 配置文件 | 分辨率 | 畸变 |
|---|---|---|---|
| `data/images/` 图片 | `config/camera_image.yaml` | 960×720 | 全 0（仿真相机 rune_sim_camera） |
| `data/videos/armor_record.avi` | `config/camera_video.yaml` | 1440×1080 | 有（标定值，重投影误差 0.1448px） |

图片和视频是两套内参。混用会让误差大好几倍，而且**程序不会报错**，
只会安静地输出一个错得离谱的数。

视频内参的主点是 (679.2, 529.1)，即成像中心对应 1358×1058，与素材的 1440×1080 不同。
这个差值只改变光轴位置，不改变尺度：第三周用这套内参在视频上做 PnP，解出的装甲板
半径为 0.207 m，与官方 /tf 真值的 0.2088 m 相差 1.5 mm，焦距可以直接使用。

---

## 5. 真值表格式（data/gt/ground_truth.csv）

```
image,armor_id,visible,tx,ty,tz
image1,1,front,0.285747,0.070019,1.384573
```

- 坐标：相机光学系，**单位米**（x 向右 / y 向下 / z 沿光轴指向物体）
- `visible=front` 的行才参与误差对比（背面朝相机的板根本拍不到）
- 程序按**三维距离最近**给检测结果配真值编号（阈值 1000mm）

> 为什么不用"图像位置就近"：图像上位置近 ≠ 三维位置近。实测 image3 中检出的板，
> 图像中心离真值 armor_3（背面，z=1.553m）只有 31px，离 armor_2（正面，z=1.282m）却有 59px，
> 按图像就近会配到 armor_3，但其三维误差 333mm、armor_2 只有 105mm —— 配错了。
> 根源是深度歧义：只要 (x/z, y/z) 比值接近，投影位置就接近，而 z 可以差很多。

### 图片与真值编号的错位（已修正）

三路独立证据（灯条位置最近邻打分、PnP 解算坐标最近邻、以及"每组恰好用上一次"的完整性）
都指向同一个结论 —— 材料给的原始表 image 列与实物图片编号**整体错位了一格**：

| 图片 | 原始表同组真值的 z | 实测解算 z | 判断 |
|---|---|---|---|
| image1.jpg | 1.28 ~ 1.39 m | **4.32 m** | 差 3 倍，原始表不可能对 |
| image2.jpg | 3.76 ~ 4.09 m | **1.42 m** | 差 2.7 倍，原始表不可能对 |
| image3.jpg | 1.47 ~ 1.55 m | **1.22 m** | 差 20%，由前两组互证确定 |

`data/gt/ground_truth.csv` 已经是修正后的表（程序默认读它）；
材料原件存为 `data/gt/ground_truth_original.csv`，用来对照看错位的影响。
完整证据见 `data/gt/README.md`。

---

## 6. 装甲板尺寸与特征点定义（容易踩的坑）

```
物体坐标系（原点在装甲板中心，单位毫米）：

      (-67.5,-27.5) ┌────────┐ (67.5,-27.5)      ← 上边 = 两灯条上端连线
                    │        │
                    └────────┘                    ← 下边 = 两灯条下端连线
      (-67.5, 27.5)            (67.5, 27.5)

宽 135mm = 两灯条**中心距**，高 55mm = 灯条长度
```

这两个值必须和检测器给出的角点定义配套。数值取自官方《机器人制作规范手册》的小装甲
模块 135×55：本项目 `image_points` 用的是两根灯条的真实上下端点，所以 135mm 取
两灯条中心距、55mm 取灯条长度，与开源项目的通行口径一致。

按实测反推，本素材的等效中心距为 143~145 mm，比规格值 135 mm 大 6%，这是深度方向
系统偏差的主要来源。代码保留官方规格 135 mm，不用实测值反调。

---

## 7. 已知差异

`visible` 标记的是装甲板有没有被车体遮挡，能检出几块取决于灯条的颜色分割结果，
两者不是同一个条件。修正后 image1 正面可见的是 armor_3、armor_4 两块，实际检出 1 块。
程序按三维距离匹配真值编号，检出数量少于标记数量不影响误差统计。

图片与真值编号的错位见第 5 节，装甲板 135×55 的定义见第 6 节，两项均按实测证据与
官方规格确定。

---

## 8. 材料来源

- 视频：群文件 `2026-05-20_23-57-31.avi` → 已复制为 `data/videos/armor_record.avi`
- 图片真值：`图1.md / 图2.md / 图3.md` → 已整理为 `data/gt/ground_truth.csv`
- 装甲板检测网络：https://github.com/SZURPVision/26_NNDeployment_Lib_and_Detection_Models
- 数字识别网络：https://github.com/chenjunnn/rm_auto_aim/tree/main/armor_detector
  （`model/mlp.onnx` 即数字识别网络，类别名见 `model/label.txt`）
- 传统视觉流程参考了 `chenjunnn/rm_auto_aim` 的 `armor_detector`
  （预处理 → findLights → matchLights → PnP），具体差异写在
  `include/traditional_detector.h` 的注释里。
