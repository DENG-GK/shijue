# Sobel算子

> Sobel算子是最经典、最常用的边缘检测算子之一。它巧妙地结合了**平滑**和**求导**两个操作，既能检测边缘，又能抑制噪声。掌握Sobel算子，是学习边缘检测的第一步！

---

## 📖 理论部分

### 1. Sobel算子的起源与特点

#### 1.1 为什么需要Sobel？

在上一节中，我们学习了最简单的差分核来检测边缘。但简单差分有一个问题：

```
简单差分的问题：

  简单差分核：              实际图像往往有噪声：
  ┌────┬────┬────┐
  │ -1 │ 0  │ 1  │          理想边缘        实际边缘（有噪声）
  └────┴────┴────┘
                              │              │ ∧
  只看一行像素                │              │∧│∧
  对噪声非常敏感！            │──────        │┴─┴──
                                            噪声被当成边缘！

  解决思路：
  在求导的同时，做一些平滑处理，抑制噪声的影响
```

Sobel算子正是这个思路的产物！

#### 1.2 Sobel算子的设计思想

```
Sobel算子的巧妙设计：

  将平滑和求导结合在一起！

  Sobel核 = 平滑核 × 差分核

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  水平方向 Sobel (Gx)：                                       │
  │                                                             │
  │   平滑（垂直方向）     差分（水平方向）      Sobel Gx        │
  │   ┌───┐              ┌───┬───┬───┐      ┌────┬────┬────┐   │
  │   │ 1 │              │-1 │ 0 │ 1 │      │ -1 │ 0  │ +1 │   │
  │   ├───┤      ×       └───┴───┴───┘  =   ├────┼────┼────┤   │
  │   │ 2 │                                 │ -2 │ 0  │ +2 │   │
  │   ├───┤                                 ├────┼────┼────┤   │
  │   │ 1 │                                 │ -1 │ 0  │ +1 │   │
  │   └───┘                                 └────┴────┴────┘   │
  │                                                             │
  │  垂直方向类似...                                             │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  中间权重为2的原因：
  • 加强当前行（列）的贡献
  • 相当于 [1,2,1] 的平滑核（近似高斯）
  • 在保持导数精度的同时提供平滑效果
```

---

### 2. Sobel算子的数学原理

#### 2.1 Sobel卷积核

```
Sobel算子的两个卷积核：

  Gx（水平方向梯度）            Gy（垂直方向梯度）
  用于检测【垂直边缘】          用于检测【水平边缘】

  ┌────┬────┬────┐              ┌────┬────┬────┐
  │ -1 │ 0  │ +1 │              │ -1 │ -2 │ -1 │
  ├────┼────┼────┤              ├────┼────┼────┤
  │ -2 │ 0  │ +2 │              │ 0  │ 0  │ 0  │
  ├────┼────┼────┤              ├────┼────┼────┤
  │ -1 │ 0  │ +1 │              │ +1 │ +2 │ +1 │
  └────┴────┴────┘              └────┴────┴────┘

  记忆技巧：
  Gx：左负右正，检测左右变化（垂直边缘）
  Gy：上负下正，检测上下变化（水平边缘）

  为什么Gx检测的是垂直边缘？

    想象一条垂直的边缘：
    ┌─────────────┐
    │ ██│        │
    │ ██│        │   左边深，右边浅
    │ ██│        │   水平方向（x）变化大
    │ ██│        │   所以Gx响应大！
    └─────────────┘
```

#### 2.2 卷积计算过程

```
Sobel卷积的计算示例：

  假设图像局部区域是：
  ┌─────┬─────┬─────┐
  │ 10  │ 10  │ 100 │
  ├─────┼─────┼─────┤
  │ 10  │ 10  │ 100 │    （左边暗，右边亮，有垂直边缘）
  ├─────┼─────┼─────┤
  │ 10  │ 10  │ 100 │
  └─────┴─────┴─────┘

  应用 Gx 核：
  ┌────┬────┬────┐
  │ -1 │ 0  │ +1 │
  ├────┼────┼────┤
  │ -2 │ 0  │ +2 │
  ├────┼────┼────┤
  │ -1 │ 0  │ +1 │
  └────┴────┴────┘

  Gx = (-1)×10 + 0×10 + (+1)×100
     + (-2)×10 + 0×10 + (+2)×100
     + (-1)×10 + 0×10 + (+1)×100

     = -10 + 100 - 20 + 200 - 10 + 100
     = 360 （大正值！说明有从暗到亮的垂直边缘）

  应用 Gy 核：
  Gy = (-1)×10 + (-2)×10 + (-1)×100
     + 0×10 + 0×10 + 0×100
     + (+1)×10 + (+2)×10 + (+1)×100

     = -10 - 20 - 100 + 10 + 20 + 100
     = 0 （没有水平边缘）

  梯度幅值 = √(360² + 0²) = 360
  说明这里确实有一条垂直边缘！
```

#### 2.3 梯度幅值与方向

```
从Gx和Gy计算最终结果：

  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  梯度幅值（边缘强度）：                                 │
  │                                                        │
  │    精确公式：|G| = √(Gx² + Gy²)                        │
  │                                                        │
  │    近似公式：|G| ≈ |Gx| + |Gy|  （计算更快）           │
  │                                                        │
  │  梯度方向（边缘方向的垂直方向）：                       │
  │                                                        │
  │    θ = arctan(Gy / Gx)                                 │
  │                                                        │
  └────────────────────────────────────────────────────────┘

  可视化理解：

                    Gy
                    ↑
                    │   ╱ G（梯度向量）
                    │  ╱
                    │ ╱  |G| = √(Gx² + Gy²)
                    │╱ θ = arctan(Gy/Gx)
       ─────────────┼───────────→ Gx
                    │

  注意：梯度方向垂直于边缘方向！
  如果检测到的梯度方向是水平的，说明边缘是垂直的
```

---

### 3. OpenCV中的Sobel函数

#### 3.1 函数语法

```
cv2.Sobel() 函数详解：

  dst = cv2.Sobel(src, ddepth, dx, dy, ksize=3, scale=1, delta=0)

  参数说明：
  ┌────────────┬─────────────────────────────────────────────────┐
  │ 参数       │ 说明                                             │
  ├────────────┼─────────────────────────────────────────────────┤
  │ src        │ 输入图像（通常是灰度图）                         │
  ├────────────┼─────────────────────────────────────────────────┤
  │ ddepth     │ 输出图像深度（重要！）                           │
  │            │ • cv2.CV_64F：64位浮点，保留负值                 │
  │            │ • cv2.CV_32F：32位浮点                           │
  │            │ • cv2.CV_16S：16位有符号整数                     │
  │            │ • -1：与输入相同（不推荐）                       │
  ├────────────┼─────────────────────────────────────────────────┤
  │ dx         │ x方向求导阶数（0或1）                            │
  ├────────────┼─────────────────────────────────────────────────┤
  │ dy         │ y方向求导阶数（0或1）                            │
  ├────────────┼─────────────────────────────────────────────────┤
  │ ksize      │ 核大小（1, 3, 5, 7，或-1使用Scharr）             │
  ├────────────┼─────────────────────────────────────────────────┤
  │ scale      │ 缩放因子（默认1）                                │
  ├────────────┼─────────────────────────────────────────────────┤
  │ delta      │ 加到结果上的偏移量（默认0）                      │
  └────────────┴─────────────────────────────────────────────────┘

  常用调用方式：
  # 计算x方向梯度（检测垂直边缘）
  sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

  # 计算y方向梯度（检测水平边缘）
  sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
```

#### 3.2 关于ddepth的重要说明

```
为什么ddepth很重要？

  问题：导数可能是负数！

  考虑这种情况（从亮到暗的边缘）：
  ┌─────┬─────┬─────┐
  │ 200 │ 200 │ 50  │
  ├─────┼─────┼─────┤
  │ 200 │ 200 │ 50  │   Gx = -300（负值！）
  ├─────┼─────┼─────┤
  │ 200 │ 200 │ 50  │
  └─────┴─────┴─────┘

  如果使用 uint8 类型（只能存0-255）：
  -300 会被截断为 0！→ 丢失了这条边缘！

  正确做法：
  1. 使用 cv2.CV_64F 或 cv2.CV_32F 保留负值
  2. 取绝对值
  3. 再转换为 uint8 显示

  代码示例：
  sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # 保留负值
  sobel_x = np.abs(sobel_x)                    # 取绝对值
  sobel_x = sobel_x.astype(np.uint8)           # 转换类型
```

---

### 4. 不同ksize的影响

```
ksize参数的影响：

  ksize=1（实际是1×3或3×1）：
  ┌────┬────┬────┐
  │ -1 │ 0  │ 1  │    最简单的差分，无平滑
  └────┴────┴────┘    对噪声敏感

  ksize=3（标准Sobel）：
  ┌────┬────┬────┐
  │ -1 │ 0  │ +1 │
  ├────┼────┼────┤    有一定平滑效果
  │ -2 │ 0  │ +2 │    抗噪声能力适中
  ├────┼────┼────┤
  │ -1 │ 0  │ +1 │
  └────┴────┴────┘

  ksize=5：
  ┌────┬────┬────┬────┬────┐
  │ -1 │ -2 │ 0  │ +2 │ +1 │
  ├────┼────┼────┼────┼────┤
  │ -4 │ -8 │ 0  │ +8 │ +4 │
  ├────┼────┼────┼────┼────┤    更强的平滑
  │ -6 │-12 │ 0  │+12 │ +6 │    更好的抗噪声能力
  ├────┼────┼────┼────┼────┤    但边缘可能变粗
  │ -4 │ -8 │ 0  │ +8 │ +4 │
  ├────┼────┼────┼────┼────┤
  │ -1 │ -2 │ 0  │ +2 │ +1 │
  └────┴────┴────┴────┴────┘

  ksize=-1（使用Scharr核）：
  ┌────┬────┬────┐
  │ -3 │ 0  │ +3 │
  ├────┼────┼────┤    精度更高
  │-10 │ 0  │+10 │    尤其是对角方向
  ├────┼────┼────┤    见下一节详解
  │ -3 │ 0  │ +3 │
  └────┴────┴────┘

  选择建议：
  ┌──────────┬────────────────────────────────────┐
  │ ksize=3  │ 默认选择，适合大多数情况           │
  ├──────────┼────────────────────────────────────┤
  │ ksize=5  │ 图像噪声较多时                     │
  ├──────────┼────────────────────────────────────┤
  │ ksize=-1 │ 需要高精度时（使用Scharr）         │
  └──────────┴────────────────────────────────────┘
```

---

### 5. Sobel的优缺点

```
Sobel算子的优缺点分析：

  ✅ 优点：
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. 计算简单、速度快                                          │
  │    - 只需要卷积运算                                          │
  │    - 实时处理完全可行                                        │
  │                                                             │
  │ 2. 自带平滑效果                                              │
  │    - [1,2,1] 的加权平滑                                     │
  │    - 比简单差分抗噪声                                        │
  │                                                             │
  │ 3. 可分离计算                                                │
  │    - 3×3的核可以分解为两个1×3的核                           │
  │    - 进一步提高计算效率                                      │
  │                                                             │
  │ 4. 方向信息                                                  │
  │    - 分别计算Gx和Gy                                         │
  │    - 可以得到边缘的方向                                      │
  └─────────────────────────────────────────────────────────────┘

  ❌ 缺点：
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. 边缘较粗                                                  │
  │    - 不如Canny的边缘细                                       │
  │    - 可能需要后续细化处理                                    │
  │                                                             │
  │ 2. 对角线方向精度较低                                        │
  │    - 3×3 Sobel对45度边缘响应不够好                          │
  │    - 可以用Scharr改善                                        │
  │                                                             │
  │ 3. 仍然对噪声有一定敏感性                                    │
  │    - 虽然比简单差分好                                        │
  │    - 但不如Canny的效果                                       │
  │                                                             │
  │ 4. 需要后处理                                                │
  │    - 需要合并Gx和Gy                                         │
  │    - 可能需要二值化                                          │
  └─────────────────────────────────────────────────────────────┘
```

---

## 💻 代码实战

### 代码1：基本的Sobel边缘检测

```python
"""
Sobel边缘检测基础用法
学习如何使用cv2.Sobel()进行边缘检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建包含各种边缘的测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 100  # 灰色背景

    # 矩形（有水平和垂直边缘）
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)

    # 圆形（有各个方向的边缘）
    cv2.circle(img, (280, 100), 50, 200, -1)

    # 斜线（有对角边缘）
    cv2.line(img, (50, 200), (180, 280), 200, 5)

    # 渐变区域
    for i in range(100):
        img[200:280, 220+i] = 100 + i

    return img

# 使用测试图像
img = create_test_image()

# 如果想用真实图像：
# img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

print("图像信息：")
print(f"  尺寸: {img.shape}")
print(f"  数据类型: {img.dtype}")

# ===================== 应用Sobel算子 =====================

# 计算x方向梯度（检测垂直边缘）
# 注意：使用CV_64F保留负值！
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# 计算y方向梯度（检测水平边缘）
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

print(f"\nSobel结果信息：")
print(f"  sobel_x 范围: [{sobel_x.min():.1f}, {sobel_x.max():.1f}]")
print(f"  sobel_y 范围: [{sobel_y.min():.1f}, {sobel_y.max():.1f}]")

# ===================== 处理结果 =====================

# 方法1：取绝对值
sobel_x_abs = np.abs(sobel_x)
sobel_y_abs = np.abs(sobel_y)

# 方法2：使用convertScaleAbs（推荐）
sobel_x_cv = cv2.convertScaleAbs(sobel_x)
sobel_y_cv = cv2.convertScaleAbs(sobel_y)

# 计算梯度幅值
# 方法1：精确计算
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

# 方法2：近似计算（更快）
magnitude_approx = cv2.addWeighted(sobel_x_cv, 0.5, sobel_y_cv, 0.5, 0)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

# x方向梯度
axes[0, 1].imshow(sobel_x_cv, cmap='gray')
axes[0, 1].set_title('Sobel X（垂直边缘）', fontsize=12)
axes[0, 1].axis('off')

# y方向梯度
axes[0, 2].imshow(sobel_y_cv, cmap='gray')
axes[0, 2].set_title('Sobel Y（水平边缘）', fontsize=12)
axes[0, 2].axis('off')

# 梯度幅值（精确）
axes[1, 0].imshow(magnitude, cmap='gray')
axes[1, 0].set_title('梯度幅值（精确）\n√(Gx²+Gy²)', fontsize=12)
axes[1, 0].axis('off')

# 梯度幅值（近似）
axes[1, 1].imshow(magnitude_approx, cmap='gray')
axes[1, 1].set_title('梯度幅值（近似）\n|Gx|+|Gy|', fontsize=12)
axes[1, 1].axis('off')

# 带颜色的方向可视化
axes[1, 2].imshow(sobel_x, cmap='RdBu', vmin=-200, vmax=200)
axes[1, 2].set_title('Sobel X（带正负）\n红=正，蓝=负', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle('Sobel边缘检测基础', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_basic.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图像已保存为 'sobel_basic.png'")
```

---

### 代码2：ddepth参数的影响

```python
"""
演示ddepth参数对Sobel结果的影响
说明为什么要使用CV_64F而不是uint8
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_edge_image():
    """创建一个有明暗两种边缘的图像"""
    img = np.zeros((200, 300), dtype=np.uint8)
    img[:] = 128  # 中等灰度背景

    # 左边：从亮到暗的边缘（导数为负）
    img[:, 50:100] = 200  # 亮区域

    # 右边：从暗到亮的边缘（导数为正）
    img[:, 200:250] = 200  # 亮区域

    return img

img = create_edge_image()

print("测试图像说明：")
print("  左边：亮→暗（x=100处，梯度为负）")
print("  右边：暗→亮（x=200处，梯度为正）")

# ===================== 不同ddepth的Sobel =====================

# 1. 使用uint8（错误做法）
# 注意：这会导致负值被截断为0！
sobel_uint8 = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)

# 2. 使用CV_64F（正确做法）
sobel_float = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# 3. 对float结果取绝对值
sobel_abs = np.abs(sobel_float)
sobel_abs = np.clip(sobel_abs, 0, 255).astype(np.uint8)

# 4. 使用convertScaleAbs
sobel_scale = cv2.convertScaleAbs(sobel_float)

print(f"\n不同方法的结果范围：")
print(f"  CV_8U直接计算: [{sobel_uint8.min()}, {sobel_uint8.max()}]")
print(f"  CV_64F计算:    [{sobel_float.min():.1f}, {sobel_float.max():.1f}]")
print(f"  取绝对值后:    [{sobel_abs.min()}, {sobel_abs.max()}]")

# ===================== 可视化对比 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像\n左:亮→暗  右:暗→亮', fontsize=11)
axes[0, 0].axis('off')

# 一维剖面图
row_data = img[100, :]
axes[0, 1].plot(row_data, 'b-', linewidth=2)
axes[0, 1].axvline(x=100, color='r', linestyle='--', alpha=0.5)
axes[0, 1].axvline(x=200, color='g', linestyle='--', alpha=0.5)
axes[0, 1].set_title('中间行的灰度值分布', fontsize=11)
axes[0, 1].set_xlabel('像素位置')
axes[0, 1].set_ylabel('灰度值')
axes[0, 1].grid(True, alpha=0.3)

# CV_8U结果（错误）
axes[0, 2].imshow(sobel_uint8, cmap='gray')
axes[0, 2].set_title('❌ 错误：CV_8U\n左边的边缘丢失了！', fontsize=11, color='red')
axes[0, 2].axis('off')

# CV_64F原始结果
im = axes[1, 0].imshow(sobel_float, cmap='RdBu', vmin=-300, vmax=300)
axes[1, 0].set_title('CV_64F原始结果\n红=正，蓝=负', fontsize=11)
axes[1, 0].axis('off')
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# 取绝对值后
axes[1, 1].imshow(sobel_abs, cmap='gray')
axes[1, 1].set_title('✓ 正确：CV_64F + 取绝对值\n两边的边缘都检测到了', fontsize=11, color='green')
axes[1, 1].axis('off')

# 对比一维剖面
row_uint8 = sobel_uint8[100, :]
row_abs = sobel_abs[100, :]
axes[1, 2].plot(row_uint8, 'r-', linewidth=2, label='CV_8U（错误）')
axes[1, 2].plot(row_abs, 'g-', linewidth=2, label='CV_64F+abs（正确）')
axes[1, 2].axvline(x=100, color='gray', linestyle='--', alpha=0.5)
axes[1, 2].axvline(x=200, color='gray', linestyle='--', alpha=0.5)
axes[1, 2].set_title('Sobel结果对比', fontsize=11)
axes[1, 2].set_xlabel('像素位置')
axes[1, 2].set_ylabel('梯度幅值')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('ddepth参数的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_ddepth.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n结论：")
print("  使用CV_8U会丢失负方向的边缘！")
print("  正确做法：使用CV_64F，然后取绝对值")
print("\n图像已保存为 'sobel_ddepth.png'")
```

---

### 代码3：不同ksize的效果对比

```python
"""
对比不同ksize参数对Sobel边缘检测的影响
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建带噪声的测试图像 =====================

def create_noisy_image():
    """创建一个带噪声的图像"""
    # 基础图像
    img = np.zeros((250, 350), dtype=np.uint8)
    img[:] = 80

    # 添加形状
    cv2.rectangle(img, (50, 50), (150, 150), 180, -1)
    cv2.circle(img, (250, 100), 50, 180, -1)

    # 斜线
    cv2.line(img, (50, 180), (150, 230), 180, 4)

    # 添加高斯噪声
    noise = np.random.normal(0, 15, img.shape).astype(np.float64)
    noisy = img.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return img, noisy

clean_img, noisy_img = create_noisy_image()

print("测试图像：")
print(f"  干净图像")
print(f"  带噪声图像（高斯噪声，标准差=15）")

# ===================== 不同ksize的Sobel =====================

ksize_list = [1, 3, 5, 7]
results = {}

for ksize in ksize_list:
    # 对噪声图像应用Sobel
    sobel_x = cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=ksize)

    # 计算梯度幅值
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    results[ksize] = magnitude

# 也测试ksize=-1（Scharr）
sobel_x_scharr = cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=-1)
sobel_y_scharr = cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=-1)
magnitude_scharr = np.sqrt(sobel_x_scharr**2 + sobel_y_scharr**2)
magnitude_scharr = np.clip(magnitude_scharr, 0, 255).astype(np.uint8)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 第一行：原图和ksize 1, 3
axes[0, 0].imshow(clean_img, cmap='gray')
axes[0, 0].set_title('原始图像（无噪声）', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(noisy_img, cmap='gray')
axes[0, 1].set_title('带噪声图像', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(results[1], cmap='gray')
axes[0, 2].set_title('ksize=1\n（最简单，噪声明显）', fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(results[3], cmap='gray')
axes[0, 3].set_title('ksize=3\n（标准Sobel）', fontsize=11)
axes[0, 3].axis('off')

# 第二行：ksize 5, 7, Scharr和说明
axes[1, 0].imshow(results[5], cmap='gray')
axes[1, 0].set_title('ksize=5\n（更多平滑）', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(results[7], cmap='gray')
axes[1, 1].set_title('ksize=7\n（最多平滑，边缘变粗）', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(magnitude_scharr, cmap='gray')
axes[1, 2].set_title('ksize=-1 (Scharr)\n（更精确）', fontsize=11)
axes[1, 2].axis('off')

# 说明文字
axes[1, 3].axis('off')
info_text = """
ksize 选择指南：

ksize=1:
  • 最简单的差分
  • 对噪声非常敏感
  • 很少使用

ksize=3:
  • 标准Sobel核
  • 适合大多数场景
  • 推荐作为默认选择

ksize=5:
  • 更强的平滑效果
  • 抗噪声能力更好
  • 边缘可能略粗

ksize=7:
  • 最强的平滑
  • 边缘会变粗
  • 适合噪声很大的图像

ksize=-1 (Scharr):
  • 精度比3×3 Sobel高
  • 对角方向效果更好
  • 适合需要精确检测时
"""
axes[1, 3].text(0.1, 0.5, info_text, fontsize=9,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('不同ksize对Sobel边缘检测的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_ksize.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图像已保存为 'sobel_ksize.png'")
```

---

### 代码4：Sobel检测不同方向的边缘

```python
"""
演示Sobel算子如何分别检测不同方向的边缘
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建有各种方向边缘的图像 =====================

def create_direction_test():
    """创建测试图像，包含不同方向的边缘"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 100

    # 垂直边缘
    cv2.rectangle(img, (30, 50), (80, 250), 200, -1)

    # 水平边缘
    cv2.rectangle(img, (120, 80), (280, 130), 200, -1)

    # 斜线（45度）
    cv2.line(img, (120, 170), (220, 270), 200, 8)

    # 斜线（-45度）
    cv2.line(img, (250, 170), (350, 270), 200, 8)

    # 圆形（各方向边缘）
    cv2.circle(img, (320, 80), 40, 200, -1)

    return img

img = create_direction_test()

# ===================== 分别检测各方向边缘 =====================

# X方向梯度（检测垂直边缘）
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_x_abs = cv2.convertScaleAbs(sobel_x)

# Y方向梯度（检测水平边缘）
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# 合并两个方向
sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

# 精确幅值
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

# 计算梯度方向
direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

print("边缘检测结果：")
print("  Sobel X: 检测垂直边缘")
print("  Sobel Y: 检测水平边缘")
print("  合并结果: 所有方向的边缘")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像\n包含垂直、水平、斜向边缘', fontsize=11)
axes[0, 0].axis('off')

# X方向
axes[0, 1].imshow(sobel_x_abs, cmap='gray')
axes[0, 1].set_title('Sobel X（|Gx|）\n检测垂直边缘 ↑↓', fontsize=11)
axes[0, 1].axis('off')

# Y方向
axes[0, 2].imshow(sobel_y_abs, cmap='gray')
axes[0, 2].set_title('Sobel Y（|Gy|）\n检测水平边缘 ←→', fontsize=11)
axes[0, 2].axis('off')

# 加权合并
axes[1, 0].imshow(sobel_combined, cmap='gray')
axes[1, 0].set_title('加权合并\n0.5×|Gx| + 0.5×|Gy|', fontsize=11)
axes[1, 0].axis('off')

# 精确幅值
axes[1, 1].imshow(magnitude, cmap='gray')
axes[1, 1].set_title('精确幅值\n√(Gx² + Gy²)', fontsize=11)
axes[1, 1].axis('off')

# 梯度方向可视化
# 只在边缘处显示方向
mask = magnitude > 30
direction_masked = np.where(mask, direction, 0)
im = axes[1, 2].imshow(direction_masked, cmap='hsv', vmin=-180, vmax=180)
axes[1, 2].set_title('梯度方向\n（颜色表示角度）', fontsize=11)
axes[1, 2].axis('off')
plt.colorbar(im, ax=axes[1, 2], fraction=0.046, label='角度 (度)')

plt.suptitle('Sobel算子检测不同方向的边缘', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_directions.png', dpi=150, bbox_inches='tight')
plt.show()

# 额外：打印不同边缘的梯度方向
print("\n梯度方向说明：")
print("  垂直边缘: Gx大, Gy≈0 → 方向接近0°或180°")
print("  水平边缘: Gx≈0, Gy大 → 方向接近90°或-90°")
print("  45°斜边: Gx≈Gy → 方向接近45°或-135°")
print("\n图像已保存为 'sobel_directions.png'")
```

---

### 代码5：Sobel边缘检测完整流程

```python
"""
Sobel边缘检测的完整处理流程
包括预处理、边缘检测、后处理
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建或读取图像 =====================

def create_sample_image():
    """创建一个模拟的实际场景图像"""
    img = np.zeros((350, 450, 3), dtype=np.uint8)

    # 背景渐变
    for y in range(350):
        img[y, :] = [100 + y//5, 100 + y//5, 100 + y//5]

    # 建筑物
    cv2.rectangle(img, (50, 100), (180, 300), (80, 80, 120), -1)
    cv2.rectangle(img, (70, 120), (100, 180), (150, 150, 100), -1)  # 窗户
    cv2.rectangle(img, (130, 120), (160, 180), (150, 150, 100), -1)  # 窗户
    cv2.rectangle(img, (100, 220), (140, 300), (60, 60, 80), -1)  # 门

    # 树
    cv2.rectangle(img, (240, 200), (270, 300), (50, 80, 60), -1)  # 树干
    cv2.circle(img, (255, 150), 60, (40, 100, 50), -1)  # 树冠

    # 汽车
    cv2.rectangle(img, (300, 250), (420, 300), (60, 60, 150), -1)  # 车身
    cv2.circle(img, (330, 300), 20, (30, 30, 30), -1)  # 轮子
    cv2.circle(img, (390, 300), 20, (30, 30, 30), -1)  # 轮子

    # 添加一些噪声使其更真实
    noise = np.random.normal(0, 10, img.shape).astype(np.float64)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

# 创建图像
color_img = create_sample_image()

# 如果使用真实图像：
# color_img = cv2.imread('your_image.jpg')

# ===================== 步骤1：预处理 =====================

# 1.1 转换为灰度图
gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

# 1.2 降噪（高斯模糊）
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

print("步骤1：预处理完成")
print(f"  灰度图尺寸: {gray.shape}")

# ===================== 步骤2：Sobel边缘检测 =====================

# 2.1 计算X方向梯度
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)

# 2.2 计算Y方向梯度
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# 2.3 计算梯度幅值
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

print("步骤2：Sobel边缘检测完成")

# ===================== 步骤3：后处理 =====================

# 3.1 二值化
_, binary = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

# 3.2 形态学操作（可选，细化边缘）
kernel = np.ones((2, 2), np.uint8)
morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 3.3 边缘叠加到原图
edges_colored = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
edges_colored[morphed > 0] = [0, 255, 0]  # 绿色边缘
overlay = cv2.addWeighted(color_img, 0.7, edges_colored, 0.3, 0)

print("步骤3：后处理完成")

# ===================== 可视化完整流程 =====================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 第一行：处理流程
axes[0, 0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('1. 原始彩色图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('2. 灰度转换', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(blurred, cmap='gray')
axes[0, 2].set_title('3. 高斯模糊降噪', fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(magnitude, cmap='gray')
axes[0, 3].set_title('4. Sobel边缘检测', fontsize=11)
axes[0, 3].axis('off')

# 第二行：后处理和结果
axes[1, 0].imshow(cv2.convertScaleAbs(sobel_x), cmap='gray')
axes[1, 0].set_title('Sobel X（垂直边缘）', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.convertScaleAbs(sobel_y), cmap='gray')
axes[1, 1].set_title('Sobel Y（水平边缘）', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(binary, cmap='gray')
axes[1, 2].set_title('5. 二值化', fontsize=11)
axes[1, 2].axis('off')

axes[1, 3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title('6. 边缘叠加到原图', fontsize=11)
axes[1, 3].axis('off')

plt.suptitle('Sobel边缘检测完整流程', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_complete_pipeline.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n完整流程图已保存为 'sobel_complete_pipeline.png'")
```

---

### 代码6：Sobel实际应用 - 车道线检测简化版

```python
"""
Sobel边缘检测的实际应用：简化的车道线检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建模拟的道路图像 =====================

def create_road_image():
    """创建一个模拟的道路图像"""
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # 天空（上半部分）
    img[:200, :] = [180, 200, 220]

    # 道路（下半部分）
    for y in range(200, 400):
        img[y, :] = [80, 80, 80]

    # 车道线（白色虚线）
    # 左车道线
    for i in range(0, 200, 40):
        y1 = 200 + i
        y2 = min(200 + i + 25, 399)
        x1 = 200 - i
        x2 = 200 - i - 15
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)

    # 右车道线
    for i in range(0, 200, 40):
        y1 = 200 + i
        y2 = min(200 + i + 25, 399)
        x1 = 400 + i
        x2 = 400 + i + 15
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)

    # 中心线（黄色实线）
    cv2.line(img, (300, 200), (300, 400), (0, 200, 200), 3)

    # 添加一些噪声
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

road_img = create_road_image()

print("模拟道路图像已创建")

# ===================== 车道线检测流程 =====================

# 1. 转换为灰度图
gray = cv2.cvtColor(road_img, cv2.COLOR_BGR2GRAY)

# 2. ROI（感兴趣区域）- 只关注道路部分
roi = gray[200:, :]  # 下半部分

# 3. 高斯模糊
blurred = cv2.GaussianBlur(roi, (5, 5), 0)

# 4. Sobel边缘检测
# 车道线主要是垂直方向的，所以重点检测X方向梯度
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_x_abs = np.abs(sobel_x)
sobel_x_abs = np.clip(sobel_x_abs, 0, 255).astype(np.uint8)

# 也检测Y方向（用于完整性）
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_y_abs = np.abs(sobel_y)
sobel_y_abs = np.clip(sobel_y_abs, 0, 255).astype(np.uint8)

# 合并
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

# 5. 二值化
_, binary = cv2.threshold(sobel_x_abs, 80, 255, cv2.THRESH_BINARY)

# 6. 形态学处理
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

# 7. 将检测结果放回原图大小
result_full = np.zeros_like(gray)
result_full[200:, :] = cleaned

# 8. 标记检测到的车道线
result_color = road_img.copy()
result_color[result_full > 0] = [0, 255, 0]  # 绿色标记

print("车道线检测完成")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 第一行
axes[0, 0].imshow(cv2.cvtColor(road_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('1. 原始道路图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(roi, cmap='gray')
axes[0, 1].set_title('2. ROI区域（道路部分）', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(blurred, cmap='gray')
axes[0, 2].set_title('3. 高斯模糊', fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(sobel_x_abs, cmap='gray')
axes[0, 3].set_title('4. Sobel X（检测垂直线）', fontsize=11)
axes[0, 3].axis('off')

# 第二行
axes[1, 0].imshow(magnitude, cmap='gray')
axes[1, 0].set_title('5. 梯度幅值', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(binary, cmap='gray')
axes[1, 1].set_title('6. 二值化', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(cleaned, cmap='gray')
axes[1, 2].set_title('7. 形态学清理', fontsize=11)
axes[1, 2].axis('off')

axes[1, 3].imshow(cv2.cvtColor(result_color, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title('8. 检测结果（绿色标记）', fontsize=11)
axes[1, 3].axis('off')

plt.suptitle('Sobel边缘检测应用：车道线检测', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_lane_detection.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n车道线检测结果已保存为 'sobel_lane_detection.png'")
print("\n说明：")
print("  这是一个简化的车道线检测示例")
print("  实际应用中还需要：")
print("  - 透视变换（鸟瞰图）")
print("  - 霍夫变换（直线检测）")
print("  - 曲线拟合")
print("  - 时序滤波等")
```

---

## 📝 本节总结

```
┌────────────────────────────────────────────────────────────────────┐
│                        Sobel算子总结                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. Sobel算子的设计                                                 │
│     • 结合平滑和求导：Sobel = 平滑核 × 差分核                       │
│     • 中间权重为2，加强当前行/列的贡献                              │
│     • 相比简单差分，抗噪声能力更好                                  │
│                                                                    │
│  2. Sobel卷积核                                                     │
│                                                                    │
│     Gx（检测垂直边缘）         Gy（检测水平边缘）                   │
│     ┌────┬────┬────┐          ┌────┬────┬────┐                    │
│     │ -1 │ 0  │ +1 │          │ -1 │ -2 │ -1 │                    │
│     │ -2 │ 0  │ +2 │          │ 0  │ 0  │ 0  │                    │
│     │ -1 │ 0  │ +1 │          │ +1 │ +2 │ +1 │                    │
│     └────┴────┴────┘          └────┴────┴────┘                    │
│                                                                    │
│  3. OpenCV函数                                                      │
│     dst = cv2.Sobel(src, ddepth, dx, dy, ksize=3)                  │
│                                                                    │
│  4. 重要注意事项                                                    │
│     • ddepth使用CV_64F，保留负值                                   │
│     • 处理后取绝对值再转uint8                                      │
│     • ksize=3是标准选择，噪声大时用5或7                            │
│     • ksize=-1使用Scharr核，精度更高                               │
│                                                                    │
│  5. 梯度幅值计算                                                    │
│     • 精确：|G| = √(Gx² + Gy²)                                     │
│     • 近似：|G| ≈ |Gx| + |Gy|                                      │
│                                                                    │
│  6. 处理流程                                                        │
│     灰度转换 → 降噪 → Sobel检测 → 合并Gx和Gy → 后处理              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 下一步学习

在下一节 **"03-Scharr算子"** 中，我们将：
- 学习Scharr算子的原理（Sobel的改进版）
- 了解Scharr为什么比Sobel更精确
- 对比Sobel和Scharr在不同场景下的效果

---

> 💡 **学习建议**：Sobel是最基础、最常用的边缘检测算子。建议多练习调整ksize参数，观察对不同图像的效果差异！
