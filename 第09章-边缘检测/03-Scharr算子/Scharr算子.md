# Scharr算子

> Scharr算子是Sobel算子的"升级版"。虽然Sobel已经很好用了，但它在检测某些方向（特别是对角线）的边缘时精度不够高。Scharr算子通过优化卷积核的权重，实现了更好的旋转对称性和更高的检测精度！

---

## 📖 理论部分

### 1. 为什么需要Scharr？

#### 1.1 Sobel的局限性

```
Sobel算子的问题：旋转对称性不够好

  什么是旋转对称性？
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  理想情况：无论边缘方向如何，检测响应应该一致                │
  │                                                             │
  │  例如：同样强度的边缘，不管是水平、垂直还是45度，           │
  │       梯度幅值应该相同                                      │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  Sobel的问题：

  使用3×3 Sobel检测不同角度的边缘：

    0°（水平边缘）     45°（对角边缘）     90°（垂直边缘）
    ┌───────────┐     ┌───────────┐      ┌───────────┐
    │           │     │      ╱    │      │  │        │
    │ ───────── │     │    ╱      │      │  │        │
    │           │     │  ╱        │      │  │        │
    └───────────┘     └───────────┘      └───────────┘
    响应：100%         响应：~70%?         响应：100%
                            ↑
                       问题出在这里！
                       45度边缘响应偏低

  原因分析：
  Sobel核的设计在0°和90°方向是精确的
  但在45°方向会有约30%的误差！
```

#### 1.2 Scharr的解决方案

```
Scharr核的设计目标：

  优化卷积核权重，使得在所有方向上的响应更加一致

  设计原则：
  1. 保持可分离性（计算效率）
  2. 保持反对称性（导数性质）
  3. 最小化各向异性误差（旋转对称性）

  通过数学优化得到最佳权重：

  Sobel:  [-1, 0, 1]^T × [1, 2, 1]    权重比 1:2
  Scharr: [-1, 0, 1]^T × [3, 10, 3]   权重比 3:10

  Scharr的权重不是随意选择的，而是通过最小化角度误差
  求解得到的最优解！
```

---

### 2. Scharr卷积核

#### 2.1 核的结构

```
Scharr算子的卷积核：

  Gx（检测垂直边缘）            Gy（检测水平边缘）

  ┌────┬────┬────┐              ┌────┬────┬────┐
  │ -3 │ 0  │ +3 │              │ -3 │-10 │ -3 │
  ├────┼────┼────┤              ├────┼────┼────┤
  │-10 │ 0  │+10 │              │ 0  │ 0  │ 0  │
  ├────┼────┼────┤              ├────┼────┼────┤
  │ -3 │ 0  │ +3 │              │ +3 │+10 │ +3 │
  └────┴────┴────┘              └────┴────┴────┘

  对比Sobel：

  Sobel Gx:                     Scharr Gx:
  ┌────┬────┬────┐              ┌────┬────┬────┐
  │ -1 │ 0  │ +1 │              │ -3 │ 0  │ +3 │
  ├────┼────┼────┤              ├────┼────┼────┤
  │ -2 │ 0  │ +2 │              │-10 │ 0  │+10 │
  ├────┼────┼────┤              ├────┼────┼────┤
  │ -1 │ 0  │ +1 │              │ -3 │ 0  │ +3 │
  └────┴────┴────┘              └────┴────┴────┘
  权重比 1:2:1                   权重比 3:10:3

  Scharr中间行的权重是角落的 10/3 ≈ 3.33 倍
  而Sobel只有 2/1 = 2 倍
```

#### 2.2 为什么3:10:3更好？

```
直观理解Scharr的权重选择：

  计算导数时，我们希望：
  • 中间行（当前位置）的贡献最大
  • 上下行提供平滑，但贡献应该适当

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  Sobel (1:2:1)：                                            │
  │  • 中间权重是角落的2倍                                       │
  │  • 平滑效果适中                                              │
  │  • 对角方向误差较大                                          │
  │                                                             │
  │  Scharr (3:10:3)：                                          │
  │  • 中间权重是角落的3.33倍                                    │
  │  • 更强调当前行的贡献                                        │
  │  • 对角方向误差最小化                                        │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  数学上的解释：
  3:10:3 这个比例是通过求解一个优化问题得到的：

    minimize: 角度误差
    subject to:
      - 核可分离
      - 核反对称
      - 归一化约束

  3:10:3 是这个优化问题的最优解！
```

---

### 3. Sobel vs Scharr 对比

#### 3.1 精度对比

```
精度对比（理论分析）：

  对于单位阶跃边缘，不同角度的响应：

  角度    Sobel响应    Scharr响应    理想响应
  ────────────────────────────────────────────
   0°      1.000        1.000        1.000
  15°      0.966        0.991        1.000
  30°      0.866        0.962        1.000
  45°      0.707        0.918        1.000  ← 最大差异处
  60°      0.866        0.962        1.000
  75°      0.966        0.991        1.000
  90°      1.000        1.000        1.000

  可以看到：
  • 在45度方向，Sobel误差约30%，Scharr误差约8%
  • Scharr的角度误差比Sobel小约4倍！

  图形化表示（各向同性对比）：

     Sobel                    Scharr                   理想
       ↑                        ↑                       ↑
       │   * *                  │  *****                │ *****
       │  *   *                 │ **   **               │**   **
       │ *     *                │*       *              │*     *
    ←──┼──────→             ←──┼────────→           ←──┼──────→
       │ *     *                │*       *              │*     *
       │  *   *                 │ **   **               │**   **
       │   * *                  │  *****                │ *****
       ↓                        ↓                       ↓
    略呈方形                   更接近圆形               完美圆形
```

#### 3.2 使用场景对比

```
何时使用Sobel vs Scharr？

  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  使用 Sobel (ksize=3) 的场景：                                │
  │  ────────────────────────────                                │
  │  • 一般的边缘检测任务                                         │
  │  • 对精度要求不高                                             │
  │  • 需要更大的平滑效果（可用ksize=5,7）                        │
  │  • 计算资源有限时                                             │
  │                                                              │
  │  使用 Scharr 的场景：                                         │
  │  ────────────────────                                        │
  │  • 需要高精度边缘检测                                         │
  │  • 图像中有大量斜向边缘                                       │
  │  • 后续需要精确的梯度方向                                     │
  │  • 工业视觉、测量等精密应用                                   │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  实际差异示例：

  场景：检测45度的细线

  Sobel结果：          Scharr结果：
  ┌──────────┐         ┌──────────┐
  │    ╱╱    │         │    ╱     │
  │   ╱╱     │         │   ╱      │
  │  ╱╱      │         │  ╱       │
  │ ╱╱       │         │ ╱        │
  └──────────┘         └──────────┘
  边缘略宽              边缘更细更准

  注意：对于大多数日常应用，这个差异可能不明显
  但在精密测量或科学研究中，这个差异可能很关键！
```

---

### 4. OpenCV中的Scharr

#### 4.1 使用方法

```
OpenCV中使用Scharr的两种方式：

  方式1：使用 cv2.Scharr() 函数
  ─────────────────────────────

  dst = cv2.Scharr(src, ddepth, dx, dy, scale=1, delta=0)

  参数说明：
  ┌────────────┬───────────────────────────────────────────────┐
  │ 参数       │ 说明                                           │
  ├────────────┼───────────────────────────────────────────────┤
  │ src        │ 输入图像                                       │
  ├────────────┼───────────────────────────────────────────────┤
  │ ddepth     │ 输出深度（推荐CV_64F或CV_32F）                 │
  ├────────────┼───────────────────────────────────────────────┤
  │ dx         │ x方向求导阶数（0或1）                          │
  ├────────────┼───────────────────────────────────────────────┤
  │ dy         │ y方向求导阶数（0或1）                          │
  ├────────────┼───────────────────────────────────────────────┤
  │ scale      │ 缩放因子（默认1）                              │
  ├────────────┼───────────────────────────────────────────────┤
  │ delta      │ 偏移量（默认0）                                │
  └────────────┴───────────────────────────────────────────────┘

  注意：cv2.Scharr() 没有ksize参数，因为Scharr核固定是3×3

  示例：
  scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)  # X方向
  scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)  # Y方向


  方式2：使用 cv2.Sobel() + ksize=-1
  ──────────────────────────────────

  当ksize设为-1时，Sobel函数会使用Scharr核：

  scharr_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
  scharr_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

  这两种方式完全等效！
```

#### 4.2 完整使用示例

```python
# Scharr边缘检测的标准用法

import cv2
import numpy as np

# 读取图像
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 方法1：使用cv2.Scharr()
scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)

# 方法2：等效于cv2.Sobel(..., ksize=-1)
# scharr_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
# scharr_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

# 计算梯度幅值
magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

# 显示结果
cv2.imshow('Scharr Edge', magnitude)
cv2.waitKey(0)
```

---

### 5. Scharr的优缺点

```
Scharr算子的优缺点：

  ✅ 优点：
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  1. 更高的精度                                               │
  │     • 角度误差比Sobel小约4倍                                 │
  │     • 对角方向边缘检测更准确                                 │
  │                                                             │
  │  2. 更好的旋转对称性                                         │
  │     • 各方向响应更一致                                       │
  │     • 梯度幅值计算更准确                                     │
  │                                                             │
  │  3. 梯度方向更准确                                           │
  │     • 对于需要方向信息的应用很重要                           │
  │     • 如霍夫变换、边缘跟踪等                                 │
  │                                                             │
  │  4. 计算量与3×3 Sobel相同                                   │
  │     • 只是权重不同，不增加计算成本                           │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  ❌ 缺点：
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  1. 只有3×3尺寸                                             │
  │     • 不像Sobel可以有5×5、7×7等大尺寸                       │
  │     • 无法增加平滑程度                                       │
  │                                                             │
  │  2. 对噪声相对敏感                                           │
  │     • 因为只有3×3，平滑能力有限                             │
  │     • 需要配合预滤波使用                                     │
  │                                                             │
  │  3. 改进效果有限                                             │
  │     • 在大多数实际应用中                                     │
  │     • 与Sobel的差异可能不明显                                │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  总结：
  如果精度很重要 → 用Scharr
  如果需要更多平滑 → 用Sobel (ksize=5,7)
  一般情况 → 两者差别不大，可以都试试
```

---

## 💻 代码实战

### 代码1：Scharr基本用法

```python
"""
Scharr算子的基本用法
学习如何使用cv2.Scharr()进行边缘检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建包含各种方向边缘的测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 100

    # 垂直边缘（0°）
    cv2.rectangle(img, (30, 50), (80, 250), 200, -1)

    # 水平边缘（90°）
    cv2.rectangle(img, (110, 80), (220, 130), 200, -1)

    # 45度斜线
    cv2.line(img, (110, 170), (200, 260), 200, 8)

    # -45度斜线
    cv2.line(img, (230, 170), (320, 260), 200, 8)

    # 圆形（各方向边缘）
    cv2.circle(img, (320, 100), 45, 200, -1)

    return img

img = create_test_image()

print("测试图像信息：")
print(f"  尺寸: {img.shape}")

# ===================== 应用Scharr算子 =====================

# 方法1：使用cv2.Scharr()
scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)  # X方向
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)  # Y方向

# 方法2：等效方法 - cv2.Sobel(..., ksize=-1)
# scharr_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
# scharr_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

print(f"\nScharr结果：")
print(f"  scharr_x 范围: [{scharr_x.min():.1f}, {scharr_x.max():.1f}]")
print(f"  scharr_y 范围: [{scharr_y.min():.1f}, {scharr_y.max():.1f}]")

# ===================== 处理结果 =====================

# 取绝对值
scharr_x_abs = cv2.convertScaleAbs(scharr_x)
scharr_y_abs = cv2.convertScaleAbs(scharr_y)

# 计算梯度幅值
magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

# 计算梯度方向
direction = np.arctan2(scharr_y, scharr_x) * 180 / np.pi

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像\n(各种方向的边缘)', fontsize=11)
axes[0, 0].axis('off')

# Scharr X
axes[0, 1].imshow(scharr_x_abs, cmap='gray')
axes[0, 1].set_title('Scharr X\n(检测垂直边缘)', fontsize=11)
axes[0, 1].axis('off')

# Scharr Y
axes[0, 2].imshow(scharr_y_abs, cmap='gray')
axes[0, 2].set_title('Scharr Y\n(检测水平边缘)', fontsize=11)
axes[0, 2].axis('off')

# 梯度幅值
axes[1, 0].imshow(magnitude, cmap='gray')
axes[1, 0].set_title('梯度幅值\n√(Gx² + Gy²)', fontsize=11)
axes[1, 0].axis('off')

# 带颜色的方向
mask = magnitude > 30
direction_masked = np.where(mask, direction, np.nan)
im = axes[1, 1].imshow(direction_masked, cmap='hsv', vmin=-180, vmax=180)
axes[1, 1].set_title('梯度方向\n(颜色表示角度)', fontsize=11)
axes[1, 1].axis('off')
plt.colorbar(im, ax=axes[1, 1], fraction=0.046, label='角度(度)')

# 说明
axes[1, 2].axis('off')
info = """
Scharr算子使用说明：

1. 两种等效的调用方式：
   • cv2.Scharr(img, ddepth, dx, dy)
   • cv2.Sobel(img, ddepth, dx, dy, ksize=-1)

2. 参数说明：
   • ddepth: 使用CV_64F保留负值
   • dx=1, dy=0: 检测垂直边缘
   • dx=0, dy=1: 检测水平边缘

3. 后处理：
   • 取绝对值：cv2.convertScaleAbs()
   • 梯度幅值：√(Gx² + Gy²)

4. Scharr只有3×3尺寸
   如需更大核，请使用Sobel
"""
axes[1, 2].text(0.1, 0.5, info, fontsize=10,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Scharr边缘检测基础', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('scharr_basic.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图像已保存为 'scharr_basic.png'")
```

---

### 代码2：Sobel vs Scharr 精度对比

```python
"""
对比Sobel和Scharr在不同角度边缘上的检测精度
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建不同角度的边缘 =====================

def create_angled_edge(angle_deg, size=100):
    """创建指定角度的边缘图像"""
    img = np.zeros((size, size), dtype=np.uint8)

    # 使用旋转矩阵创建斜线
    center = size // 2
    angle_rad = np.radians(angle_deg)

    # 创建一条通过中心的线
    for i in range(size):
        for j in range(size):
            # 计算点到线的距离
            x = j - center
            y = i - center
            # 旋转坐标
            x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)
            if x_rot > 0:
                img[i, j] = 200
            else:
                img[i, j] = 50

    return img

# 创建不同角度的边缘图像
angles = [0, 15, 30, 45, 60, 75, 90]
edge_images = {angle: create_angled_edge(angle) for angle in angles}

print("创建了不同角度的边缘图像：", angles)

# ===================== 计算响应 =====================

def compute_max_gradient(img, method='sobel'):
    """计算最大梯度幅值"""
    if method == 'sobel':
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    else:  # scharr
        gx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(img, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(gx**2 + gy**2)

    # 返回中心区域的最大值（避免边界效应）
    center_region = magnitude[30:70, 30:70]
    return np.max(center_region)

# 计算不同角度的响应
sobel_responses = []
scharr_responses = []

for angle in angles:
    img = edge_images[angle]
    sobel_responses.append(compute_max_gradient(img, 'sobel'))
    scharr_responses.append(compute_max_gradient(img, 'scharr'))

# 归一化（以0度为基准）
sobel_norm = [r / sobel_responses[0] for r in sobel_responses]
scharr_norm = [r / scharr_responses[0] for r in scharr_responses]

print("\n归一化响应对比（以0°为100%）：")
print(f"{'角度':>6} | {'Sobel':>8} | {'Scharr':>8} | {'理想':>8}")
print("-" * 40)
for i, angle in enumerate(angles):
    print(f"{angle:>5}° | {sobel_norm[i]*100:>7.1f}% | {scharr_norm[i]*100:>7.1f}% | {100.0:>7.1f}%")

# ===================== 可视化 =====================

fig = plt.figure(figsize=(16, 10))

# 上半部分：不同角度的边缘图像
for i, angle in enumerate(angles):
    ax = plt.subplot(3, 7, i+1)
    ax.imshow(edge_images[angle], cmap='gray')
    ax.set_title(f'{angle}°', fontsize=10)
    ax.axis('off')

# 中间：Sobel检测结果
for i, angle in enumerate(angles):
    ax = plt.subplot(3, 7, 7+i+1)
    img = edge_images[angle]
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    ax.imshow(mag, cmap='gray')
    ax.set_title(f'Sobel\n{sobel_norm[i]*100:.1f}%', fontsize=9)
    ax.axis('off')

# 下半部分：Scharr检测结果
for i, angle in enumerate(angles):
    ax = plt.subplot(3, 7, 14+i+1)
    img = edge_images[angle]
    gx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    ax.imshow(mag, cmap='gray')
    ax.set_title(f'Scharr\n{scharr_norm[i]*100:.1f}%', fontsize=9)
    ax.axis('off')

plt.suptitle('Sobel vs Scharr 不同角度响应对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_vs_scharr_angles.png', dpi=150, bbox_inches='tight')
plt.show()

# 绘制响应曲线
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(angles, [s*100 for s in sobel_norm], 'b-o', linewidth=2, markersize=8, label='Sobel')
ax.plot(angles, [s*100 for s in scharr_norm], 'r-s', linewidth=2, markersize=8, label='Scharr')
ax.axhline(y=100, color='g', linestyle='--', linewidth=2, label='理想响应')

ax.set_xlabel('边缘角度 (度)', fontsize=12)
ax.set_ylabel('归一化响应 (%)', fontsize=12)
ax.set_title('Sobel vs Scharr 角度响应曲线', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(angles)
ax.set_ylim([60, 110])

# 标注45度处的差异
ax.annotate(f'Sobel在45°: {sobel_norm[3]*100:.1f}%\nScharr在45°: {scharr_norm[3]*100:.1f}%',
            xy=(45, sobel_norm[3]*100), xytext=(55, 75),
            fontsize=10, fontfamily='SimHei',
            arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('sobel_vs_scharr_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n对比图已保存")
```

---

### 代码3：实际图像上的Sobel vs Scharr对比

```python
"""
在实际图像上对比Sobel和Scharr的效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_complex_image():
    """创建一个复杂的测试图像"""
    img = np.zeros((350, 450), dtype=np.uint8)
    img[:] = 80

    # 建筑物（有各种角度的边缘）
    # 主体
    pts = np.array([[100, 280], [100, 120], [200, 60], [300, 120], [300, 280]], np.int32)
    cv2.fillPoly(img, [pts], 180)

    # 窗户
    cv2.rectangle(img, (130, 150), (170, 200), 100, -1)
    cv2.rectangle(img, (230, 150), (270, 200), 100, -1)

    # 门
    cv2.rectangle(img, (180, 210), (220, 280), 60, -1)

    # 圆形装饰
    cv2.circle(img, (200, 100), 20, 220, -1)

    # 斜线装饰
    cv2.line(img, (50, 300), (120, 250), 200, 3)
    cv2.line(img, (330, 250), (400, 300), 200, 3)

    # 曲线
    for i in range(100):
        x = 350 + i
        y = int(150 + 30 * np.sin(i * 0.1))
        if 0 <= x < 450 and 0 <= y < 350:
            cv2.circle(img, (x, y), 2, 200, -1)

    return img

img = create_complex_image()

# 高斯模糊预处理
img_blur = cv2.GaussianBlur(img, (3, 3), 0)

print("测试图像已创建")

# ===================== Sobel边缘检测 =====================

sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)

# ===================== Scharr边缘检测 =====================

scharr_x = cv2.Scharr(img_blur, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img_blur, cv2.CV_64F, 0, 1)
scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)
scharr_mag = np.clip(scharr_mag, 0, 255).astype(np.uint8)

# ===================== 计算差异 =====================

# 归一化后计算差异
sobel_norm = sobel_mag.astype(np.float64) / max(1, sobel_mag.max())
scharr_norm = scharr_mag.astype(np.float64) / max(1, scharr_mag.max())
diff = np.abs(sobel_norm - scharr_norm)
diff_display = (diff * 255).astype(np.uint8)

print(f"\n差异统计：")
print(f"  最大差异: {diff.max()*100:.1f}%")
print(f"  平均差异: {diff.mean()*100:.1f}%")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

# Sobel结果
axes[0, 1].imshow(sobel_mag, cmap='gray')
axes[0, 1].set_title('Sobel边缘检测', fontsize=12)
axes[0, 1].axis('off')

# Scharr结果
axes[0, 2].imshow(scharr_mag, cmap='gray')
axes[0, 2].set_title('Scharr边缘检测', fontsize=12)
axes[0, 2].axis('off')

# 差异图
im = axes[1, 0].imshow(diff_display, cmap='hot')
axes[1, 0].set_title('差异图（热力图）\n亮色=差异大', fontsize=12)
axes[1, 0].axis('off')
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# 局部放大对比（斜边区域）
region = (50, 150, 200, 300)  # (x1, y1, x2, y2)
sobel_crop = sobel_mag[region[1]:region[3], region[0]:region[2]]
scharr_crop = scharr_mag[region[1]:region[3], region[0]:region[2]]

axes[1, 1].imshow(sobel_crop, cmap='gray')
axes[1, 1].set_title('Sobel局部放大\n（斜边区域）', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(scharr_crop, cmap='gray')
axes[1, 2].set_title('Scharr局部放大\n（斜边区域）', fontsize=12)
axes[1, 2].axis('off')

# 在原图上标注放大区域
rect = plt.Rectangle((region[0], region[1]), region[2]-region[0], region[3]-region[1],
                       fill=False, edgecolor='red', linewidth=2)
axes[0, 0].add_patch(rect)

plt.suptitle('Sobel vs Scharr 实际效果对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_vs_scharr_real.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n对比图已保存为 'sobel_vs_scharr_real.png'")
```

---

### 代码4：Scharr用于精确梯度方向计算

```python
"""
演示Scharr在计算精确梯度方向时的优势
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建已知角度的边缘 =====================

def create_known_angle_image(true_angle, size=200):
    """创建具有精确已知角度的边缘"""
    img = np.zeros((size, size), dtype=np.uint8)

    center = size // 2
    angle_rad = np.radians(true_angle)

    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center
            # 点在线的哪一侧
            side = x * np.cos(angle_rad) + y * np.sin(angle_rad)
            if side > 0:
                img[i, j] = 200
            else:
                img[i, j] = 50

    return img

# 测试不同的真实角度
true_angles = [0, 22.5, 45, 67.5, 90]

results = []

for true_angle in true_angles:
    img = create_known_angle_image(true_angle)

    # 使用Sobel计算梯度方向
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # 使用Scharr计算梯度方向
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_direction = np.arctan2(scharr_y, scharr_x) * 180 / np.pi

    # 在边缘区域取平均方向
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_mask = sobel_mag > 100

    sobel_mean = np.mean(sobel_direction[edge_mask])
    scharr_mean = np.mean(scharr_direction[edge_mask])

    # 梯度方向垂直于边缘方向，所以需要+90°
    expected_gradient = true_angle + 90
    if expected_gradient > 180:
        expected_gradient -= 360

    # 计算误差
    sobel_error = abs(sobel_mean - expected_gradient)
    if sobel_error > 180:
        sobel_error = 360 - sobel_error
    scharr_error = abs(scharr_mean - expected_gradient)
    if scharr_error > 180:
        scharr_error = 360 - scharr_error

    results.append({
        'true_angle': true_angle,
        'expected_gradient': expected_gradient,
        'sobel_direction': sobel_mean,
        'scharr_direction': scharr_mean,
        'sobel_error': sobel_error,
        'scharr_error': scharr_error,
        'image': img
    })

# ===================== 打印结果 =====================

print("梯度方向精度对比：")
print("=" * 70)
print(f"{'边缘角度':>10} | {'期望梯度':>10} | {'Sobel':>10} | {'Scharr':>10} | {'Sobel误差':>10} | {'Scharr误差':>10}")
print("-" * 70)
for r in results:
    print(f"{r['true_angle']:>9}° | {r['expected_gradient']:>9.1f}° | {r['sobel_direction']:>9.1f}° | "
          f"{r['scharr_direction']:>9.1f}° | {r['sobel_error']:>9.1f}° | {r['scharr_error']:>9.1f}°")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 5, figsize=(16, 7))

for i, r in enumerate(results):
    # 上排：原图
    axes[0, i].imshow(r['image'], cmap='gray')
    axes[0, i].set_title(f"边缘角度: {r['true_angle']}°", fontsize=10)
    axes[0, i].axis('off')

    # 下排：误差柱状图
    errors = [r['sobel_error'], r['scharr_error']]
    bars = axes[1, i].bar(['Sobel', 'Scharr'], errors, color=['blue', 'red'])
    axes[1, i].set_ylabel('方向误差 (度)')
    axes[1, i].set_title(f'误差对比', fontsize=10)
    axes[1, i].set_ylim([0, max(5, max(errors)*1.2)])

    # 在柱子上标注数值
    for bar, error in zip(bars, errors):
        axes[1, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{error:.2f}°', ha='center', va='bottom', fontsize=9)

plt.suptitle('Sobel vs Scharr 梯度方向精度对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('scharr_direction_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

# 绘制误差汇总图
fig, ax = plt.subplots(figsize=(10, 6))

angles = [r['true_angle'] for r in results]
sobel_errors = [r['sobel_error'] for r in results]
scharr_errors = [r['scharr_error'] for r in results]

x = np.arange(len(angles))
width = 0.35

bars1 = ax.bar(x - width/2, sobel_errors, width, label='Sobel', color='steelblue')
bars2 = ax.bar(x + width/2, scharr_errors, width, label='Scharr', color='coral')

ax.set_xlabel('边缘角度 (度)', fontsize=12)
ax.set_ylabel('方向误差 (度)', fontsize=12)
ax.set_title('Sobel vs Scharr 方向精度对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{a}°' for a in angles])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}°', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}°', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('scharr_error_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n结论：")
print("  Scharr在计算梯度方向时比Sobel更精确")
print("  特别是在45度附近的边缘，优势更明显")
print("\n图像已保存")
```

---

### 代码5：Scharr实际应用 - 精密测量

```python
"""
Scharr算子的实际应用：精密边缘测量
模拟工业视觉中的零件边缘检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建模拟的工业零件图像 =====================

def create_industrial_part():
    """创建一个模拟的工业零件图像"""
    img = np.zeros((400, 500), dtype=np.uint8)
    img[:] = 50  # 暗背景

    # 主体（六边形零件）
    center = (250, 200)
    radius = 120
    pts = []
    for i in range(6):
        angle = i * 60 + 15  # 倾斜15度
        x = int(center[0] + radius * np.cos(np.radians(angle)))
        y = int(center[1] + radius * np.sin(np.radians(angle)))
        pts.append([x, y])
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], 180)

    # 中心孔
    cv2.circle(img, center, 40, 50, -1)

    # 小孔
    for i in range(6):
        angle = i * 60 + 45
        x = int(center[0] + 80 * np.cos(np.radians(angle)))
        y = int(center[1] + 80 * np.sin(np.radians(angle)))
        cv2.circle(img, (x, y), 15, 50, -1)

    # 添加少量噪声（模拟真实拍摄）
    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

img = create_industrial_part()

print("工业零件图像已创建")

# ===================== 边缘检测 =====================

# 预处理
img_blur = cv2.GaussianBlur(img, (3, 3), 0)

# Sobel检测
sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

# Scharr检测
scharr_x = cv2.Scharr(img_blur, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img_blur, cv2.CV_64F, 0, 1)
scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)

# ===================== 亚像素边缘定位 =====================

def find_subpixel_edges(magnitude, threshold=50):
    """简化的亚像素边缘定位"""
    # 二值化找粗略边缘
    _, binary = cv2.threshold(magnitude.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)

    # 细化
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 找轮廓点
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 收集所有边缘点
    edge_points = []
    for contour in contours:
        for point in contour:
            edge_points.append(point[0])

    return np.array(edge_points), binary

sobel_points, sobel_binary = find_subpixel_edges(sobel_mag)
scharr_points, scharr_binary = find_subpixel_edges(scharr_mag)

print(f"\n边缘点数量：")
print(f"  Sobel: {len(sobel_points)} 个点")
print(f"  Scharr: {len(scharr_points)} 个点")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始零件图像', fontsize=12)
axes[0, 0].axis('off')

# Sobel幅值
axes[0, 1].imshow(sobel_mag, cmap='gray', vmax=300)
axes[0, 1].set_title('Sobel梯度幅值', fontsize=12)
axes[0, 1].axis('off')

# Scharr幅值
axes[0, 2].imshow(scharr_mag, cmap='gray', vmax=500)
axes[0, 2].set_title('Scharr梯度幅值', fontsize=12)
axes[0, 2].axis('off')

# Sobel边缘叠加
overlay_sobel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
overlay_sobel[sobel_binary > 0] = [0, 255, 0]
axes[1, 0].imshow(overlay_sobel)
axes[1, 0].set_title('Sobel边缘（绿色）', fontsize=12)
axes[1, 0].axis('off')

# Scharr边缘叠加
overlay_scharr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
overlay_scharr[scharr_binary > 0] = [0, 0, 255]
axes[1, 1].imshow(overlay_scharr)
axes[1, 1].set_title('Scharr边缘（红色）', fontsize=12)
axes[1, 1].axis('off')

# 边缘对比（局部放大）
# 放大六边形的一个角
region = (300, 100, 400, 200)
sobel_crop = sobel_mag[region[1]:region[3], region[0]:region[2]]
scharr_crop = scharr_mag[region[1]:region[3], region[0]:region[2]]

# 差异可视化
diff = np.abs(sobel_crop - scharr_crop)
axes[1, 2].imshow(diff, cmap='hot')
axes[1, 2].set_title('局部差异热力图\n（斜边区域）', fontsize=12)
axes[1, 2].axis('off')

# 在原图上标注放大区域
rect = plt.Rectangle((region[0], region[1]), region[2]-region[0], region[3]-region[1],
                       fill=False, edgecolor='yellow', linewidth=2)
axes[0, 0].add_patch(rect)

plt.suptitle('工业视觉：Scharr精密边缘检测', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('scharr_industrial.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n工业应用图已保存为 'scharr_industrial.png'")
print("\n说明：")
print("  在工业视觉中，精确的边缘检测对于：")
print("  • 尺寸测量")
print("  • 位置定位")
print("  • 缺陷检测")
print("  都非常重要，Scharr的高精度在这些场景中很有价值")
```

---

### 代码6：Scharr参数调节工具

```python
"""
交互式Sobel/Scharr对比工具
实时观察两种算子的差异
"""

import cv2
import numpy as np

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建测试图像"""
    img = np.zeros((400, 500), dtype=np.uint8)
    img[:] = 80

    # 各种形状和角度的边缘
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)

    # 斜线
    cv2.line(img, (200, 50), (300, 150), 200, 5)
    cv2.line(img, (200, 150), (300, 50), 200, 5)

    # 圆
    cv2.circle(img, (400, 100), 50, 200, -1)

    # 六边形
    center = (100, 280)
    pts = []
    for i in range(6):
        angle = i * 60 + 30
        x = int(center[0] + 60 * np.cos(np.radians(angle)))
        y = int(center[1] + 60 * np.sin(np.radians(angle)))
        pts.append([x, y])
    cv2.fillPoly(img, [np.array(pts, np.int32)], 200)

    # 菱形
    pts2 = np.array([[280, 220], [340, 280], [280, 340], [220, 280]], np.int32)
    cv2.fillPoly(img, [pts2], 200)

    # 曲线
    for i in range(100):
        x = 380 + i
        y = int(280 + 40 * np.sin(i * 0.1))
        if x < 500:
            cv2.circle(img, (x, y), 2, 200, -1)

    return img

# 全局变量
img = create_test_image()
use_blur = 1
threshold_val = 50

def nothing(x):
    pass

def update_display():
    """更新显示"""
    global img, use_blur, threshold_val

    # 获取参数
    use_blur = cv2.getTrackbarPos('Blur', 'Sobel vs Scharr')
    threshold_val = cv2.getTrackbarPos('Threshold', 'Sobel vs Scharr')

    # 预处理
    if use_blur:
        processed = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        processed = img.copy()

    # Sobel检测
    sobel_x = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(processed, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)

    # Scharr检测
    scharr_x = cv2.Scharr(processed, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(processed, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr_mag = np.clip(scharr_mag, 0, 255).astype(np.uint8)

    # 二值化
    _, sobel_bin = cv2.threshold(sobel_mag, threshold_val, 255, cv2.THRESH_BINARY)
    _, scharr_bin = cv2.threshold(scharr_mag, threshold_val, 255, cv2.THRESH_BINARY)

    # 差异图
    diff = cv2.absdiff(sobel_mag, scharr_mag)
    diff_colored = cv2.applyColorMap(diff * 3, cv2.COLORMAP_HOT)

    # 组合显示
    row1 = np.hstack([
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(scharr_mag, cv2.COLOR_GRAY2BGR)
    ])
    row2 = np.hstack([
        diff_colored,
        cv2.cvtColor(sobel_bin, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(scharr_bin, cv2.COLOR_GRAY2BGR)
    ])
    display = np.vstack([row1, row2])

    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display, 'Original', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, 'Sobel', (510, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, 'Scharr', (1010, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, 'Difference', (10, 430), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, 'Sobel Binary', (510, 430), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, 'Scharr Binary', (1010, 430), font, 0.7, (255, 255, 255), 2)

    cv2.imshow('Sobel vs Scharr', display)

# ===================== 创建窗口 =====================

cv2.namedWindow('Sobel vs Scharr', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sobel vs Scharr', 1500, 850)

cv2.createTrackbar('Blur', 'Sobel vs Scharr', 1, 1, nothing)
cv2.createTrackbar('Threshold', 'Sobel vs Scharr', 50, 255, nothing)

print("=" * 60)
print("Sobel vs Scharr 对比工具")
print("=" * 60)
print("\n窗口说明：")
print("  上排：原图 | Sobel幅值 | Scharr幅值")
print("  下排：差异热力图 | Sobel二值 | Scharr二值")
print("\n参数调节：")
print("  Blur: 是否使用高斯模糊预处理")
print("  Threshold: 二值化阈值")
print("\n按 'q' 或 ESC 退出")
print("=" * 60)

# ===================== 主循环 =====================

while True:
    update_display()

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()
print("\n程序已退出")
```

---

## 📝 本节总结

```
┌────────────────────────────────────────────────────────────────────┐
│                        Scharr算子总结                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. Scharr是什么？                                                  │
│     • Sobel算子的改进版                                            │
│     • 通过优化权重提高检测精度                                     │
│     • 具有更好的旋转对称性                                         │
│                                                                    │
│  2. Scharr卷积核                                                    │
│                                                                    │
│     Gx                              Gy                             │
│     ┌────┬────┬────┐              ┌────┬────┬────┐                │
│     │ -3 │ 0  │ +3 │              │ -3 │-10 │ -3 │                │
│     │-10 │ 0  │+10 │              │ 0  │ 0  │ 0  │                │
│     │ -3 │ 0  │ +3 │              │ +3 │+10 │ +3 │                │
│     └────┴────┴────┘              └────┴────┴────┘                │
│                                                                    │
│  3. Sobel vs Scharr                                                │
│     ┌──────────────┬─────────────────┬─────────────────┐          │
│     │              │     Sobel       │     Scharr      │          │
│     ├──────────────┼─────────────────┼─────────────────┤          │
│     │ 权重比       │     1:2:1       │     3:10:3      │          │
│     ├──────────────┼─────────────────┼─────────────────┤          │
│     │ 45°精度      │     ~70%        │     ~92%        │          │
│     ├──────────────┼─────────────────┼─────────────────┤          │
│     │ 可选尺寸     │   3,5,7等       │     仅3×3       │          │
│     └──────────────┴─────────────────┴─────────────────┘          │
│                                                                    │
│  4. OpenCV函数                                                      │
│     • cv2.Scharr(src, ddepth, dx, dy)                              │
│     • cv2.Sobel(src, ddepth, dx, dy, ksize=-1)  # 等效             │
│                                                                    │
│  5. 使用场景                                                        │
│     • 需要高精度边缘检测时用Scharr                                 │
│     • 需要更多平滑效果时用Sobel(ksize>3)                           │
│     • 工业视觉、精密测量优先选择Scharr                             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 下一步学习

在下一节 **"04-Laplacian算子"** 中，我们将：
- 学习二阶导数边缘检测的原理
- 了解Laplacian算子的特点和使用方法
- 对比一阶导数和二阶导数算子的差异

---

> 💡 **学习建议**：Scharr的优势主要体现在精度要求高的场景。日常应用中，先用Sobel，如果发现对角边缘检测不理想，再换Scharr试试！
