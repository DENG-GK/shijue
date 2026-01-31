# OpenCV 安装

> OpenCV (Open Source Computer Vision Library) 是最流行的开源计算机视觉库，提供丰富的图像处理和计算机视觉算法。

---

## 📖 理论部分

### 1. OpenCV 简介

**OpenCV** 是一个跨平台的计算机视觉和机器学习软件库。

**核心特性：**
- ✅ **2500+ 算法**：图像处理、特征检测、目标识别、机器学习等
- ✅ **多语言支持**：C++、Python、Java 等
- ✅ **跨平台**：Windows、macOS、Linux、Android、iOS
- ✅ **高性能**：C++ 实现，优化的算法
- ✅ **免费开源**：BSD 许可证

**主要功能模块：**
- 📷 图像和视频 I/O
- 🖼️ 图像处理（滤波、变换、颜色空间）
- 🔍 特征检测与匹配（SIFT、ORB、HOG）
- 🎯 目标检测与识别（Haar、HOG、DNN）
- 📹 视频分析（光流、背景分割、跟踪）
- 🧠 机器学习（SVM、决策树、神经网络）

---

### 2. OpenCV 版本说明

**主要版本：**
| 版本 | 说明 | 推荐度 |
|------|------|--------|
| **OpenCV 3.x** | 旧版本，部分项目仍在使用 | ⭐⭐ |
| **OpenCV 4.x** | 当前主流版本，性能优化 | ⭐⭐⭐⭐⭐ |

**Python 包版本：**
- **opencv-python**：标准版，包含主要模块
- **opencv-contrib-python**：扩展版，包含额外算法（SIFT、SURF 等）
- **opencv-python-headless**：无 GUI 版，适合服务器
- **opencv-contrib-python-headless**：扩展无 GUI 版

**本教程推荐：opencv-contrib-python 4.8+**

---

### 3. 安装方法

OpenCV 有多种安装方式，推荐使用 pip 安装。

**方法对比：**

| 方法 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **pip 安装** | 简单快速 | 预编译版本 | ⭐⭐⭐⭐⭐ |
| **conda 安装** | 管理依赖 | 版本可能较旧 | ⭐⭐⭐⭐ |
| **源码编译** | 自定义优化 | 复杂耗时 | ⭐⭐ |

---

### 4. 常见问题处理

**问题1：import cv2 报错**
- 检查安装是否成功
- 检查 Python 版本兼容性
- 检查虚拟环境是否正确激活

**问题2：DLL 加载失败（Windows）**
- 安装 Visual C++ Redistributable
- 检查系统位数（32位/64位）

**问题3：图像显示窗口无法打开**
- 安装标准版 opencv-python（非 headless）
- 检查显示器连接

---

## 💻 代码实战

### pip 安装 OpenCV

#### 标准安装

```bash
# 激活虚拟环境
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

# 安装标准版 OpenCV
pip install opencv-python

# 或安装扩展版（推荐）
pip install opencv-contrib-python

# 指定版本安装
pip install opencv-contrib-python==4.8.1.78

# 从清华镜像安装（加速）
pip install opencv-contrib-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**安装过程：**
```
Collecting opencv-contrib-python
  Downloading opencv_contrib_python-4.8.1.78-cp37-abi3-win_amd64.whl (44.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.7/44.7 MB 5.2 MB/s eta 0:00:00
Collecting numpy>=1.21.2
  Using cached numpy-1.24.3-cp311-cp311-win_amd64.whl (14.8 MB)
Installing collected packages: numpy, opencv-contrib-python
Successfully installed numpy-1.24.3 opencv-contrib-python-4.8.1.78
```

---

#### conda 安装

```bash
# 激活 conda 环境
conda activate cv_env

# 安装 OpenCV
conda install -c conda-forge opencv

# 或指定版本
conda install -c conda-forge opencv=4.8
```

---

### 验证安装

创建测试脚本 `test_opencv.py`：

```python
# test_opencv.py
# OpenCV 安装验证脚本

import cv2
import numpy as np

print("=" * 70)
print("🎉 OpenCV 安装验证")
print("=" * 70)

# 显示 OpenCV 版本
print(f"\n📌 OpenCV 版本：{cv2.__version__}")

# 显示编译信息
build_info = cv2.getBuildInformation()
print(f"\n📌 OpenCV 编译信息（部分）：")
for line in build_info.split('\n')[:15]:
    print(f"   {line}")

# 检查可用模块
print(f"\n📦 OpenCV 主要模块：")
modules = [
    'core', 'imgproc', 'imgcodecs', 'videoio',
    'highgui', 'video', 'features2d', 'objdetect',
    'dnn', 'ml', 'photo', 'stitching'
]

for module in modules:
    if hasattr(cv2, module):
        print(f"   ✅ {module}")
    else:
        print(f"   ❌ {module}")

# 检查扩展模块（contrib）
print(f"\n🎁 OpenCV 扩展模块（contrib）：")
contrib_modules = [
    'xfeatures2d',  # SIFT、SURF 等
    'aruco',        # ArUco 标记检测
    'tracking',     # 高级跟踪算法
]

has_contrib = False
for module in contrib_modules:
    if hasattr(cv2, module):
        print(f"   ✅ {module}")
        has_contrib = True
    else:
        print(f"   ❌ {module}")

if has_contrib:
    print("\n   ✅ 已安装 opencv-contrib-python（扩展版）")
else:
    print("\n   ⚠️  未安装扩展模块（如需 SIFT/SURF，请安装 opencv-contrib-python）")

# 创建测试图像
print(f"\n🖼️  创建测试图像...")
img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (350, 250), (0, 255, 0), 3)
cv2.putText(img, 'OpenCV Test', (80, 160), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (255, 255, 255), 3)

# 保存测试图像
cv2.imwrite('opencv_test.png', img)
print(f"   ✅ 测试图像已保存：opencv_test.png")

# 读取并验证
img_read = cv2.imread('opencv_test.png')
if img_read is not None:
    print(f"   ✅ 图像读取成功：{img_read.shape}")
else:
    print(f"   ❌ 图像读取失败")

print("\n" + "=" * 70)
print("✅ OpenCV 安装验证完成！")
print("=" * 70)
```

**运行验证：**

```bash
python test_opencv.py
```

**预期输出：**
```
======================================================================
🎉 OpenCV 安装验证
======================================================================

📌 OpenCV 版本：4.8.1

📌 OpenCV 编译信息（部分）：
   General configuration for OpenCV 4.8.1 =====================================
     Version control:               4.8.1
   ...

📦 OpenCV 主要模块：
   ✅ core
   ✅ imgproc
   ✅ imgcodecs
   ✅ videoio
   ✅ highgui
   ✅ video
   ✅ features2d
   ✅ objdetect
   ✅ dnn
   ✅ ml
   ✅ photo
   ✅ stitching

🎁 OpenCV 扩展模块（contrib）：
   ✅ xfeatures2d
   ✅ aruco
   ✅ tracking

   ✅ 已安装 opencv-contrib-python（扩展版）

🖼️  创建测试图像...
   ✅ 测试图像已保存：opencv_test.png
   ✅ 图像读取成功：(300, 400, 3)

======================================================================
✅ OpenCV 安装验证完成！
======================================================================
```

---

### 测试示例

#### 示例1：读取并显示图像

创建 `example_display.py`：

```python
# example_display.py
import cv2

# 读取图像
img = cv2.imread('opencv_test.png')

# 检查是否成功读取
if img is None:
    print("错误：无法读取图像文件")
    exit()

# 显示图像信息
print(f"图像尺寸：{img.shape}")  # (高度, 宽度, 通道数)
print(f"数据类型：{img.dtype}")  # uint8

# 显示图像
cv2.imshow('Test Image', img)
print("按任意键关闭窗口...")
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭所有窗口
```

**运行：**
```bash
python example_display.py
```

---

#### 示例2：图像处理基础

创建 `example_process.py`：

```python
# example_process.py
import cv2
import numpy as np

# 创建彩色图像
img = np.zeros((400, 600, 3), dtype=np.uint8)

# 绘制图形
cv2.rectangle(img, (50, 50), (250, 250), (255, 0, 0), -1)    # 蓝色矩形
cv2.circle(img, (450, 150), 100, (0, 255, 0), -1)             # 绿色圆形
cv2.line(img, (0, 300), (600, 300), (0, 0, 255), 5)           # 红色直线

# 添加文字
cv2.putText(img, 'OpenCV Rocks!', (150, 350),
            cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)

# 保存图像
cv2.imwrite('opencv_example.png', img)
print("图像已保存：opencv_example.png")

# 显示图像
cv2.imshow('OpenCV Example', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

#### 示例3：图像转换

创建 `example_convert.py`：

```python
# example_convert.py
import cv2

# 读取彩色图像
img = cv2.imread('opencv_test.png')

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 转换为 HSV 颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 应用高斯模糊
blur = cv2.GaussianBlur(img, (15, 15), 0)

# 边缘检测
edges = cv2.Canny(gray, 50, 150)

# 保存结果
cv2.imwrite('gray.png', gray)
cv2.imwrite('blur.png', blur)
cv2.imwrite('edges.png', edges)

print("图像处理完成！")
print("  - gray.png: 灰度图")
print("  - blur.png: 模糊图")
print("  - edges.png: 边缘检测")

# 显示结果（拼接显示）
import numpy as np

# 将边缘图转换为3通道以便拼接
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# 水平拼接
result = np.hstack([img, gray_colored, blur, edges_colored])

cv2.imshow('Original | Gray | Blur | Edges', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 🎯 总结

本节学习了 OpenCV 的完整安装流程：

✅ **了解 OpenCV**：开源计算机视觉库，2500+ 算法
✅ **选择版本**：推荐 opencv-contrib-python 4.8+
✅ **pip 安装**：简单快速，`pip install opencv-contrib-python`
✅ **验证安装**：检查版本、模块、图像读写
✅ **基础示例**：图像读取、处理、显示

**常用函数速查：**
- `cv2.imread()` - 读取图像
- `cv2.imwrite()` - 保存图像
- `cv2.imshow()` - 显示图像
- `cv2.waitKey()` - 等待按键
- `cv2.cvtColor()` - 颜色空间转换
- `cv2.GaussianBlur()` - 高斯模糊
- `cv2.Canny()` - 边缘检测

**下一步：**
👉 [05 - NumPy安装与验证](../05-NumPy安装与验证/README.md)

---

## 📚 参考资料

- [OpenCV 官网](https://opencv.org/)
- [OpenCV Python 教程](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [OpenCV PyPI](https://pypi.org/project/opencv-python/)
- [OpenCV GitHub](https://github.com/opencv/opencv)
