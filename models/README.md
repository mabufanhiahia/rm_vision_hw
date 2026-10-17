# models/ 说明

这个目录放神经网络模型文件。**当前三个文件都已就位**（不是占位文件）：

| 文件 | 大小 | 来源 |
|---|---|---|
| `digit.onnx` | 314 KB | `chenjunnn/rm_auto_aim` → `armor_detector/model/mlp.onnx` |
| `label.txt` | 37 B | 同上 → `armor_detector/model/label.txt` |
| `armor.onnx` | 6.4 MB | `SZURPVision/26_NNDeployment_Lib_and_Detection_Models` → `所有模型/onnx/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx` |

> 模型缺失**不会导致程序崩溃**：`DetectorSet::init()` 会打一条警告，
> 传统视觉方案自动降级为「只做几何配对、不校验数字」，仍然可以正常检测和解算。

## 怎么下的（GitHub 直连不通时的办法）

国内直连 `github.com` / `raw.githubusercontent.com` 经常在 TLS 阶段被重置
（`wget` 报 `GnuTLS: Error in the pull function`），可以走 jsDelivr 的 GitHub 镜像：

```bash
# 数字识别网络
curl -L -o digit.onnx \
  "https://gcore.jsdelivr.net/gh/chenjunnn/rm_auto_aim@main/armor_detector/model/mlp.onnx"
curl -L -o label.txt \
  "https://gcore.jsdelivr.net/gh/chenjunnn/rm_auto_aim@main/armor_detector/model/label.txt"

# 装甲板检测网络（路径含中文，注意整体加引号）
curl -L -o armor.onnx \
  "https://gcore.jsdelivr.net/gh/SZURPVision/26_NNDeployment_Lib_and_Detection_Models@main/所有模型/onnx/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx"
```

`cdn.jsdelivr.net` 有时也会被重置，换 `gcore.jsdelivr.net` 或 `fastly.jsdelivr.net` 试试。
如果浏览器里能开 GitHub，也可以浏览器下载后拷进来：

```bash
cp /mnt/c/Users/<你的用户名>/Downloads/Infantry-v8n-*.onnx ~/rm_vision_hw/models/armor.onnx
```

## ⚠️ armor.onnx 改过版本号（不是原始文件）

仓库里那个 `Infantry-v8n-fp16-20260726-D1.8w-B16.onnx` 是用新版 ONNX 导出的：

- `ir_version = 10`
- `opset = 22`

而本机装的 ONNX Runtime **1.17.1** 只支持到 IR 9 / opset 20，直接加载会报：

```
Load model failed ... Unsupported model IR version: 10, max supported IR version: 9
```

所以把两个版本号降了下来（**只改了这两个 varint 数值，图的节点、权重、拓扑一个字节没动，
因为 varint 长度不变，文件大小也不变**）：

```
ir_version 10 → 9
opset      22 → 19
```

这么做安全的前提：这个图用到的算子（Conv / Mul / Add / Sigmoid / Split / Concat /
Resize / Transpose / Softmax）语义在 opset 11~13 就已定型，声明成 19 不会改变解释方式。
如果换别的模型，**不要盲目照抄**——先确认它没用到 opset 20+ 的新算子。

原始文件备份在 `~/rm_vision_hw_bak_1017/armor_original_ir10.onnx`。
以后把 ONNX Runtime 升到 1.20+ 就可以换回原文件（新版支持 IR 10 / opset 22）。

## 两个模型的规格（代码里依赖这些数值）

### digit.onnx —— 数字 / 兵种识别（小型 MLP）

- 输入 `1×1×28×20`（灰度二值图，值域 0/1）
- 算子只有 `Flatten / Gemm / Relu`，所以用 **OpenCV 的 `cv::dnn`** 加载就够了，
  不需要 ONNX Runtime（见 `src/digit_classifier.cpp`）
- 输出 9 个 logits → softmax → 取最大

### armor.onnx —— 装甲板检测（YOLOv8n + 关键点，FP16）

- 输入 `1×3×480×640`（NCHW，**float32**，图首自带一个 Cast 转 fp16）
- 输出 `1×21×6300`，每个候选 21 个数：
  `[cx, cy, w, h, kx1,ky1, kx2,ky2, kx3,ky3, kx4,ky4, cls_0..cls_8]`
- 权重是 **FP16**：ONNX Runtime 在 CPU 上会自动插 Cast 转 fp32 再算，
  能跑通但比 fp32 模型慢一些。代码里已经对「输出可能是 fp16」做了兼容
  （`YOLO_detector.cpp` 里按元素类型决定用 `float` 还是 `float16_t` 读）
- 关键点顺序：`0=左上 1=左下 2=右下 3=右上`，
  与本项目 `ArmorDetection::image_points` 的 `左上→右上→右下→左下` 不同，
  后处理里必须重排，否则 PnP 会用错位的角点

## label.txt 的内容

`digit.onnx` 的输出是 9 个类别的 logits，`label.txt` 一行一个类别名，顺序即类别下标：

```
0:1   1:2   2:3   3:4   4:5   5:outpost   6:guard   7:base   8:negative
```

最后一类 `negative` 是负样本，意思是「这块区域里根本没有数字」。
本项目用它做传统方案的配对校验：几何上像一块板、但网络判为 `negative` 的候选会被丢掉。

## 关于是否把模型提交进 Git

模型文件较大，且不同队伍/赛季会换。建议：

- 体积小的（`digit.onnx`、`label.txt`）可以直接进仓库，别人 clone 后能直接跑；
- 体积大的（`armor.onnx`，6.4 MB）用 release 附件或网盘，README 里写明下载方式。

本仓库的 `.gitignore` **没有**排除 `models/`，所以 clone 下来有什么就是什么。
