# data/gt —— 装甲板位置真值

两张表，内容相同、只有 `image` 列的对应关系不同：

| 文件 | `image` 列的对应关系 | 状态 |
|---|---|---|
| `ground_truth.csv` | image1→图1.md，image2→图2.md，image3→图3.md | 学长给的原始表（**程序默认用它**） |
| `ground_truth_reordered.csv` | image1→图2.md，image2→图3.md，image3→图1.md | 实测证据支持的重排版（整体轮转一格） |

各列含义：`image` 图片名 / `armor_id` 板上编号 / `visible` 是否正面可见（front=可见，back=被车体挡住）/
`tx,ty,tz` 装甲板中心在**相机光学系**下的坐标，单位**米**。

---

## 为什么会有「重排版」

原表的 `image` 列疑似整体错位一格。先看最硬的一条证据——**图像上的像素尺寸**。

`config/camera_image.yaml` 里 fx = 869.1，装甲板宽 135 mm，所以板子在图上的像素宽度 ≈ `135 × 869.1 / 距离`：

| 距离 | 理论板宽 | 理论板高 |
|---|---|---|
| 1.28 m | 92 px | 37 px |
| 4.06 m | 29 px | 12 px |

image1.jpg 里被框住的那块板实测约 **35 × 20 px**，和 4.06 m 那一档吻合，和 1.28 m 差了 2.6 倍。
像素尺寸只由「物理尺寸 / 距离」决定，跟 PnP、畸变、角点顺序全都无关，所以这条证据不会被算法 bug 污染。

三张图对比：

| 图片 | 原表同组真值的 z | 实测解算 z | 判断 |
|---|---|---|---|
| image1.jpg | 1.28 ~ 1.39 m | **4.32 m** | 差 3 倍，原表不可能对 |
| image2.jpg | 3.76 ~ 4.09 m | **1.42 m** | 差 2.7 倍，原表不可能对 |
| image3.jpg | 1.47 ~ 1.55 m（front） | **1.22 m** | 差 20%，单独看还说得过去 |

image1、image2 已经可以互相印证是"对调"了；image3 落单的那一组正好是图1.md，于是形成
`image1↔图2、image2↔图3、image3↔图1` 的整体轮转。重排之后三张图的平均三维误差 142 mm
（image1 269 / image2 54 / image3 104 mm），重投影 1.2~1.7 px。

## 为什么标注图上 `armor_1` 的红十字在空地上

红十字是**真值投影**，不是检测结果。把原表 image1 组的两块板按内参投回图像：
armor_1 → 像素 (659, 404)，armor_2 → (487, 426)。
而 image1.jpg 里机器人在 (487, 400) 附近，于是 armor_1 那个点孤零零落在机器人右侧的空地上。
它没画错，是它背后的真值本来就不属于这张图。

## 怎么用

```bash
# 原表（默认）
./build/rm_vision_hw image data/images/image1.jpg

# 重排表
./build/rm_vision_hw image data/images/image1.jpg data/gt/ground_truth_reordered.csv
# 三张图都可以这样跑一遍，看哪张表的误差更小
```

程序按**三维距离最近**匹配真值编号（阈值 1000 mm），不依赖 `armor_id` 的数字顺序，
所以只要 `image` 列和 `visible` 列对了，编号写成几都不影响误差计算。

## 待学长确认

1. `image` 列是否确实整体错位一格。确认后把 `ground_truth_reordered.csv` 覆盖成
   `ground_truth.csv` 即可，程序默认路径不用改。
2. `visible` 列：image1 对应图2.md 时正面可见的是 armor_3、armor_4 两块，实际只检出 1 块
   （机器人左侧那块灯条被车体挡住，属正常漏检）。
3. 135 mm 是「两灯条中心距」还是「板外缘距离」——它直接决定 z 的系统误差量级。
