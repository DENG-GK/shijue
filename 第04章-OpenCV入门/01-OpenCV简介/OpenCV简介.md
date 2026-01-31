# OpenCV 简介

> OpenCV（Open Source Computer Vision Library）是全球最广泛使用的计算机视觉库。如果说 NumPy 是图像数据的"骨架"，那么 OpenCV 就是给这副骨架装上"眼睛"和"大脑"的工具——它让计算机能够真正"看见"和"理解"图像。

---

## 📖 理论部分

### 1. 什么是 OpenCV？

#### 1.1 一句话理解

**OpenCV = 计算机视觉的瑞士军刀。** 它把图像处理、目标检测、视频分析、深度学习推理等功能全部打包成一个库，你只需要 `import cv2` 就能用。

#### 1.2 发展历史

```
时间线：

1999年  Intel 发起 OpenCV 项目（初衷：推广 Intel 芯片的图像处理能力）
2000年  首个 alpha 版本发布
2006年  OpenCV 1.0 正式发布
2009年  OpenCV 2.0（引入 C++ 接口，开始支持 Python）
2015年  OpenCV 3.0（模块化重构，引入 DNN 模块）
2018年  OpenCV 4.0（移除 C 接口，全面 C++11）
至今    OpenCV 4.x 持续更新（当前主流版本）
```

#### 1.3 核心数据

- **编程语言**：C++ 核心实现，提供 Python / Java / JavaScript 接口
- **开源协议**：BSD 许可证（商业项目也可免费使用！）
- **社区规模**：GitHub 超 7 万 Star，全球数百万开发者
- **支持平台**：Windows / Linux / macOS / Android / iOS

> 💡 **为什么选择 OpenCV 而不是其他库？** 因为它是**最全面、最成熟、生态最好**的计算机视觉库。无论你查什么图像处理的教程，99% 都是用 OpenCV。学会了它，你就能和全世界的视觉开发者交流。

---

### 2. OpenCV 能做什么？

先看几个直观的例子，感受一下 OpenCV 的能力范围：

```
OpenCV 的应用领域：

┌─────────────────────────────────────────────────────────────┐
│                        OpenCV                                │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ 图像处理  │ 目标检测  │ 视频分析  │ 3D视觉   │  深度学习推理   │
│          │          │          │          │                │
│ 滤波降噪  │ 人脸检测  │ 运动追踪  │ 相机标定  │ 分类/检测/分割  │
│ 边缘检测  │ 行人检测  │ 光流估计  │ 立体匹配  │ ONNX/TF模型   │
│ 形态学    │ 特征匹配  │ 背景分离  │ 深度估计  │ GPU加速推理    │
│ 颜色变换  │ 模板匹配  │ 目标跟踪  │ 姿态估计  │ 模型量化部署   │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
```

**具体场景举例：**

| 场景 | OpenCV 功能 | 实际应用 |
|------|------------|---------|
| 人脸识别门禁 | 人脸检测 + 特征提取 | 公司/小区门禁系统 |
| 车牌识别 | 图像预处理 + 轮廓检测 + OCR | 停车场收费系统 |
| 工业质检 | 边缘检测 + 模板匹配 | 产品缺陷检测 |
| 医学影像 | 滤波 + 分割 + 形态学 | CT/MRI 图像分析 |
| 自动驾驶 | 车道检测 + 目标检测 | 辅助驾驶系统 |
| AR滤镜 | 人脸关键点 + 图像变换 | 抖音/快手特效 |

---

### 3. OpenCV 的模块架构

OpenCV 采用模块化设计，每个模块负责一类功能：

#### 3.1 核心模块（本教程重点学习）

| 模块 | 导入名 | 功能 | 本教程涉及章节 |
|------|--------|------|--------------|
| **core** | `cv2` | 基础数据结构、数组操作 | 第04-06章 |
| **imgproc** | `cv2` | 图像处理（滤波、变换、绘图） | 第07-15章 |
| **highgui** | `cv2` | 图像显示、窗口管理、用户交互 | 第04章 |
| **imgcodecs** | `cv2` | 图像读取和保存（编解码） | 第04章 |

#### 3.2 进阶模块（后续章节学习）

| 模块 | 功能 | 本教程涉及章节 |
|------|------|--------------|
| **video** | 视频分析（光流、背景分割） | 第18-19章 |
| **videoio** | 视频读写、摄像头操作 | 第17章 |
| **calib3d** | 相机标定、立体视觉 | 第25章 |
| **features2d** | 特征检测与描述（SIFT、ORB等） | 第20-22章 |
| **objdetect** | 目标检测（Haar、HOG分类器） | 第23-24章 |
| **dnn** | 深度学习模型推理 | 第26-28章 |
| **ml** | 传统机器学习（SVM、KNN等） | 第29章 |

> 💡 **不用紧张！** 你不需要一次学完所有模块。本教程会按照由浅入深的顺序，逐步带你掌握每个模块。现在只需要知道 OpenCV 有这些能力就够了。

---

### 4. 安装 OpenCV

#### 4.1 安装命令

```bash
# 基础版（包含核心功能，推荐新手使用）
pip install opencv-python

# 完整版（包含额外模块如 SIFT、SURF 等）
pip install opencv-contrib-python

# 注意：两个包不要同时安装！选一个就好
# 推荐安装 opencv-contrib-python，功能更全
```

#### 4.2 验证安装

```python
import cv2
print(cv2.__version__)   # 输出类似：4.8.1
print("OpenCV 安装成功！")
```

#### 4.3 常见安装问题

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: No module named 'cv2'` | 重新 `pip install opencv-python` |
| 同时安装了两个版本导致冲突 | `pip uninstall opencv-python opencv-contrib-python`，然后只安装一个 |
| pip 版本过旧 | 先 `pip install --upgrade pip` |
| 使用 Anaconda | `conda install -c conda-forge opencv` |

---

### 5. OpenCV 与 NumPy 的关系（重要！）

这是新手最容易忽略但**最重要**的概念：

#### 5.1 核心事实：OpenCV 的图像就是 NumPy 数组

```python
import cv2
import numpy as np

# OpenCV 读取的图像，本质上就是 NumPy 的 ndarray
img = cv2.imread("photo.jpg")
print(type(img))        # <class 'numpy.ndarray'>  ← 就是 NumPy 数组！
print(isinstance(img, np.ndarray))  # True
```

**这意味着：**
- 你在第3章学的所有 NumPy 操作（索引、切片、运算、形状操作），**全部可以直接用在 OpenCV 图像上！**
- NumPy 不是"另一个工具"，它是 OpenCV 的基础。

#### 5.2 OpenCV 图像的数据格式

```python
img = cv2.imread("photo.jpg")

# 彩色图像
print(img.shape)    # (480, 640, 3) → (高度, 宽度, 3通道)
print(img.dtype)    # uint8 → 像素值范围 0~255

# 灰度图像
gray = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)
print(gray.shape)   # (480, 640) → (高度, 宽度)
print(gray.dtype)   # uint8
```

```
OpenCV 图像的内存布局：

彩色图像 shape = (H, W, 3):
┌─────────────────────────────────────────┐
│  像素(0,0)    像素(0,1)    像素(0,2) ... │
│  [B,G,R]      [B,G,R]      [B,G,R]     │
│                                          │
│  像素(1,0)    像素(1,1)    像素(1,2) ... │
│  [B,G,R]      [B,G,R]      [B,G,R]     │
│  ...                                     │
└─────────────────────────────────────────┘

注意通道顺序是 BGR，不是 RGB！
```

#### 5.3 为什么是 BGR 而不是 RGB？

这是 OpenCV 最"反直觉"的设计，也是新手最常踩的坑：

```python
# OpenCV 读取的图像通道顺序：B(蓝) G(绿) R(红)
# 大多数其他库（PIL、Matplotlib）使用：R(红) G(绿) B(蓝)

img = cv2.imread("photo.jpg")

# 像素值的含义
pixel = img[100, 200]       # 取第100行第200列的像素
b, g, r = pixel             # 注意顺序：B在前，R在后！
print(f"蓝={b}, 绿={g}, 红={r}")

# BGR → RGB 转换（用 Matplotlib 显示时必须转！）
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 或者用 NumPy 切片
rgb_img = img[:, :, ::-1]   # 通道维度反转
```

> 💡 **历史原因：** 早期的摄像头和显示硬件使用 BGR 顺序，OpenCV 诞生于那个年代，所以沿用了 BGR。虽然现在看起来不太方便，但为了向后兼容一直保留了下来。

#### 5.4 NumPy 操作 vs OpenCV 函数

很多操作你可以用 NumPy 做，也可以用 OpenCV 做。区别在哪？

```python
import cv2
import numpy as np

img1 = np.array([[200, 100]], dtype=np.uint8)
img2 = np.array([[100, 200]], dtype=np.uint8)

# NumPy 加法：取模运算（溢出会绕回来）
print(img1 + img2)           # [[44, 44]]  → 300%256=44，溢出！

# OpenCV 加法：饱和运算（溢出截断到255）
print(cv2.add(img1, img2))   # [[255, 255]]  → 超过255就取255，安全！
```

```
NumPy vs OpenCV 加法对比：

NumPy:   200 + 100 = 300 → 300 % 256 = 44   ← 取模，结果错误
OpenCV:  200 + 100 = 300 → min(300, 255) = 255  ← 饱和，结果正确

图像处理中，应该使用 OpenCV 的饱和运算！
```

---

### 6. OpenCV-Python 的编程惯例

#### 6.1 标准导入方式

```python
import cv2                  # OpenCV，约定俗成用 cv2
import numpy as np          # NumPy，约定俗成用 np

# 这两行是所有 OpenCV-Python 程序的开头标配
```

#### 6.2 OpenCV 函数命名规则

```python
# OpenCV 的 Python 函数命名遵循以下模式：

# 1. 全小写 + 驼峰混合
cv2.imread()           # 读取图像
cv2.imwrite()          # 保存图像
cv2.imshow()           # 显示图像
cv2.cvtColor()         # 颜色空间转换
cv2.GaussianBlur()     # 高斯模糊（注意大写开头）

# 2. 常量全大写 + 下划线
cv2.IMREAD_COLOR       # 读取标志
cv2.COLOR_BGR2GRAY     # 颜色转换标志
cv2.THRESH_BINARY      # 阈值类型
```

#### 6.3 最小完整程序

一个完整的 OpenCV-Python 程序通常长这样：

```python
import cv2
import numpy as np

# 1. 读取图像
img = cv2.imread("input.jpg")

# 2. 检查是否读取成功（非常重要！）
if img is None:
    print("错误：无法读取图像，请检查路径！")
    exit()

# 3. 处理图像（这里只是示例：转灰度）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. 显示结果
cv2.imshow("Original", img)
cv2.imshow("Gray", gray)

# 5. 等待按键并关闭窗口
cv2.waitKey(0)              # 0 表示无限等待
cv2.destroyAllWindows()     # 关闭所有窗口

# 6. 保存结果（可选）
cv2.imwrite("output_gray.jpg", gray)
```

> ⚠️ **新手必读：** `img is None` 的检查非常重要！如果图像路径错误，`cv2.imread()` 不会报错，而是**静默返回 None**。后续操作就会莫名其妙地崩溃。养成检查的习惯！

---

## 💻 代码实战

### 实战练习：OpenCV 环境验证与基础探索

```python
# ===================================================================
# 实战练习：OpenCV 环境验证与基础探索
# 目标：确认安装成功，了解基本用法
# ===================================================================

import cv2
import numpy as np

# ---------- 1. 环境验证 ----------
print("=" * 55)
print("🔧 第一步：环境验证")
print("=" * 55)

print(f"  OpenCV 版本: {cv2.__version__}")
print(f"  NumPy 版本:  {np.__version__}")

# 检查是否有 contrib 模块
try:
    sift = cv2.SIFT_create()
    print("  安装版本:    opencv-contrib-python（完整版）")
except AttributeError:
    print("  安装版本:    opencv-python（基础版）")

# ---------- 2. 用 NumPy 创建图像 ----------
print("\n" + "=" * 55)
print("🎨 第二步：用 NumPy 创建图像")
print("=" * 55)

# 创建一张纯色图像（不需要读文件！）
height, width = 300, 400

# 黑色图像
black = np.zeros((height, width, 3), dtype=np.uint8)
print(f"  黑色图像: shape={black.shape}, dtype={black.dtype}")

# 纯蓝色图像（BGR 顺序！）
blue = np.zeros((height, width, 3), dtype=np.uint8)
blue[:, :, 0] = 255    # B通道=255
print(f"  蓝色图像: 像素值={blue[0, 0]}")

# 纯绿色图像
green = np.zeros((height, width, 3), dtype=np.uint8)
green[:, :, 1] = 255   # G通道=255
print(f"  绿色图像: 像素值={green[0, 0]}")

# 纯红色图像
red = np.zeros((height, width, 3), dtype=np.uint8)
red[:, :, 2] = 255     # R通道=255
print(f"  红色图像: 像素值={red[0, 0]}")

# 白色图像
white = np.ones((height, width, 3), dtype=np.uint8) * 255
print(f"  白色图像: 像素值={white[0, 0]}")

# ---------- 3. 验证 BGR 顺序 ----------
print("\n" + "=" * 55)
print("🔍 第三步：验证 BGR 通道顺序")
print("=" * 55)

# 创建一个像素，BGR=(255, 0, 0) 在 OpenCV 中是蓝色
pixel_bgr = np.array([255, 0, 0], dtype=np.uint8)
print(f"  BGR=(255,0,0) → 蓝色")
print(f"  BGR=(0,255,0) → 绿色")
print(f"  BGR=(0,0,255) → 红色")
print(f"  BGR=(255,255,255) → 白色")
print(f"  BGR=(0,0,0) → 黑色")
print(f"  BGR=(128,128,128) → 灰色")

# ---------- 4. NumPy 与 OpenCV 的关系验证 ----------
print("\n" + "=" * 55)
print("🔗 第四步：NumPy 与 OpenCV 的关系")
print("=" * 55)

# 创建一个简单的彩色图像
img = np.zeros((4, 6, 3), dtype=np.uint8)

# 用 NumPy 切片操作图像区域
img[0:2, 0:3] = [255, 0, 0]     # 左上区域 = 蓝色
img[0:2, 3:6] = [0, 255, 0]     # 右上区域 = 绿色
img[2:4, 0:3] = [0, 0, 255]     # 左下区域 = 红色
img[2:4, 3:6] = [255, 255, 255] # 右下区域 = 白色

print(f"  图像 shape: {img.shape}")
print(f"  图像 dtype: {img.dtype}")
print(f"  type(img):  {type(img)}")
print(f"  是NumPy数组? {isinstance(img, np.ndarray)}")

# 通道分离（用 NumPy 切片）
b_channel = img[:, :, 0]
g_channel = img[:, :, 1]
r_channel = img[:, :, 2]
print(f"\n  B通道:\n{b_channel}")
print(f"\n  G通道:\n{g_channel}")
print(f"\n  R通道:\n{r_channel}")

# ---------- 5. 饱和运算 vs 取模运算 ----------
print("\n" + "=" * 55)
print("⚡ 第五步：饱和运算 vs 取模运算")
print("=" * 55)

a = np.array([[200, 100, 50]], dtype=np.uint8)
b = np.array([[100, 200, 250]], dtype=np.uint8)

numpy_add = a + b                 # NumPy: 取模
opencv_add = cv2.add(a, b)        # OpenCV: 饱和

print(f"  a:          {a}")
print(f"  b:          {b}")
print(f"  NumPy加法:  {numpy_add}   (取模: 300%256=44)")
print(f"  OpenCV加法: {opencv_add}  (饱和: min(300,255)=255)")

numpy_sub = a - b                 # NumPy: 取模（负数绕回）
opencv_sub = cv2.subtract(a, b)   # OpenCV: 饱和（负数变0）

print(f"\n  NumPy减法:  {numpy_sub}   (取模: -100+256=156)")
print(f"  OpenCV减法: {opencv_sub}  (饱和: max(-100,0)=0)")

print("\n  结论：图像处理中应该使用 cv2.add()/cv2.subtract()！")

# ---------- 6. 显示图像（需要图形界面环境） ----------
print("\n" + "=" * 55)
print("🖼️ 第六步：显示图像")
print("=" * 55)

# 创建渐变图像
gradient = np.zeros((200, 400, 3), dtype=np.uint8)
for i in range(400):
    gradient[:, i] = [int(i * 255 / 399), 0, int((399 - i) * 255 / 399)]

print("  已创建渐变图像 (200x400)")
print("  如需显示，取消以下注释：")
print("  # cv2.imshow('Gradient', gradient)")
print("  # cv2.waitKey(0)")
print("  # cv2.destroyAllWindows()")

# 取消注释以下代码可以显示图像
# cv2.imshow("Blue", blue)
# cv2.imshow("Green", green)
# cv2.imshow("Red", red)
# cv2.imshow("Gradient", gradient)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("\n✅ 所有验证通过！OpenCV 环境准备就绪。")
```

**运行输出示例：**

```
=======================================================
🔧 第一步：环境验证
=======================================================
  OpenCV 版本: 4.8.1
  NumPy 版本:  1.24.3
  安装版本:    opencv-contrib-python（完整版）

=======================================================
🎨 第二步：用 NumPy 创建图像
=======================================================
  黑色图像: shape=(300, 400, 3), dtype=uint8
  蓝色图像: 像素值=[255   0   0]
  绿色图像: 像素值=[  0 255   0]
  红色图像: 像素值=[  0   0 255]
  白色图像: 像素值=[255 255 255]

=======================================================
⚡ 第五步：饱和运算 vs 取模运算
=======================================================
  a:          [[200 100  50]]
  b:          [[100 200 250]]
  NumPy加法:  [[44 44 44]]   (取模: 300%256=44)
  OpenCV加法: [[255 255 255]]  (饱和: min(300,255)=255)

  结论：图像处理中应该使用 cv2.add()/cv2.subtract()！

✅ 所有验证通过！OpenCV 环境准备就绪。
```

---

## 🚨 常见问题与易错点

### Q1: `import cv2` 报错怎么办？

```bash
# 错误信息：ModuleNotFoundError: No module named 'cv2'

# 解决步骤：
# 1. 确认 pip 指向正确的 Python 版本
python -m pip install opencv-python

# 2. 如果使用虚拟环境，确认已激活
# conda activate myenv
# pip install opencv-python

# 3. 如果之前安装过其他版本，先卸载再重装
pip uninstall opencv-python opencv-contrib-python opencv-python-headless
pip install opencv-contrib-python
```

### Q2: BGR 和 RGB 总搞混？

```python
# 记忆口诀：OpenCV 是 BGR，其他库是 RGB

# OpenCV 读取 → BGR
img_bgr = cv2.imread("photo.jpg")   # BGR 顺序

# Matplotlib 显示需要 RGB
import matplotlib.pyplot as plt
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

# 如果不转换，Matplotlib 显示的颜色就是反的！
# 蓝色会显示成红色，红色会显示成蓝色

# 快速转换的三种方式：
rgb1 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # OpenCV 函数
rgb2 = img_bgr[:, :, ::-1]                        # NumPy 切片
rgb3 = img_bgr[:, :, [2, 1, 0]]                   # NumPy 花式索引
```

### Q3: cv2.imshow() 窗口无响应 / 闪退？

```python
# 问题：显示窗口一闪就没了，或者窗口卡住无响应

# 原因：缺少 waitKey() 调用
# cv2.imshow() 只是把图像送到窗口缓冲区
# 必须调用 cv2.waitKey() 才能真正刷新显示

# ✅ 正确的写法（完整三件套）
cv2.imshow("Window", img)
cv2.waitKey(0)              # 必须有！0=无限等待按键
cv2.destroyAllWindows()     # 关闭窗口

# 在 Jupyter Notebook 中，cv2.imshow() 可能不工作
# 替代方案：使用 Matplotlib
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

### Q4: opencv-python 和 opencv-contrib-python 有什么区别？

```python
# opencv-python:         基础版，包含核心模块
# opencv-contrib-python: 完整版，包含核心 + 扩展模块

# 扩展模块包含：
# - SIFT、SURF 等专利特征算法
# - ArUco 标记检测
# - 超分辨率模块
# - 文本检测模块
# - ...

# 建议：直接安装 opencv-contrib-python，功能最全
# 注意：两个包不能同时安装，否则会冲突！
```

---

## 🎯 总结

本节学习了 OpenCV 的基础知识：

✅ **什么是 OpenCV**：全球最流行的开源计算机视觉库，C++ 核心 + Python 接口
✅ **能做什么**：图像处理、目标检测、视频分析、3D 视觉、深度学习推理
✅ **模块架构**：模块化设计（core / imgproc / highgui / dnn 等）
✅ **安装方法**：`pip install opencv-contrib-python`
✅ **与 NumPy 关系**：OpenCV 图像就是 NumPy 数组，所有 NumPy 操作都能直接使用
✅ **BGR 通道顺序**：OpenCV 使用 BGR（不是 RGB），显示时需要转换
✅ **饱和运算**：`cv2.add()` 比 `+` 更安全（不会溢出）

> 🔑 **核心要点：** OpenCV 图像 = NumPy 数组。记住 BGR 通道顺序，养成检查 `img is None` 的习惯，使用 `cv2.add()` 代替 `+` 做图像加法。这三点是 OpenCV 入门的关键！

**下一步：**
👉 [02 - 图像读取与显示](../02-图像读取与显示/图像读取与显示.md)

---

## 📚 参考资料

- [OpenCV 官方文档](https://docs.opencv.org/)
- [OpenCV-Python 教程](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [OpenCV GitHub 仓库](https://github.com/opencv/opencv)
