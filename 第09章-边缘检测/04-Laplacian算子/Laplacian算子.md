# Laplacian算子

> Laplacian（拉普拉斯）算子是一种**二阶导数**边缘检测算子。与Sobel/Scharr不同的是，它不区分方向，一次运算就能检测到**所有方向**的边缘。但它也有一个缺点：对噪声非常敏感！

---

## 📖 理论部分

### 1. 为什么需要二阶导数？

#### 1.1 一阶导数 vs 二阶导数

```
一阶导数和二阶导数检测边缘的区别：

  原始信号（一个阶跃边缘）：

  灰度值
    │
  200├───────────────┐
    │               │
    │               │
    │               │
   50├               └──────────────
    │
    └────────────────────────────→ 位置

  一阶导数（梯度）：

    │
    │        ╱╲
    │       ╱  ╲
    │      ╱    ╲
    │─────╱      ╲─────────────
    │
    └────────────────────────────→ 位置
              ↑
         峰值位置
        （边缘位置）

  二阶导数（拉普拉斯）：

    │     ╱╲
    │    ╱  ╲
    │   ╱    ╲
    │──╱──────╳───────────────
    │         ╲    ╱
    │          ╲  ╱
    │           ╲╱
    └────────────────────────────→ 位置
              ↑
          过零点
        （边缘位置）

  关键区别：
  • 一阶导数：边缘在峰值处（最大值）
  • 二阶导数：边缘在过零点处（zero-crossing）
```

#### 1.2 二阶导数的优势

```
二阶导数检测边缘的优势：

  1. 精确定位
  ───────────
  过零点比峰值更容易精确定位
  峰值可能是一个平台，过零点是一个精确的点

  一阶导数峰值：              二阶导数过零点：
        ╱────╲                      ╱
       ╱      ╲                    ╱
      ╱        ╲                  ╱
  ───╱          ╲───         ────╳────
                                  ╲
    哪个点是峰值？               ╲
    不好确定                   这个点就是边缘！

  2. 检测所有方向
  ─────────────
  一次运算就能检测所有方向的边缘
  不需要像Sobel那样分别计算Gx和Gy再合并

  3. 双边缘效应（可以是优点也可以是缺点）
  ───────────────────────────────────────
  对于"线条"类型的边缘，会产生双边缘
  可以用于检测细线条
```

#### 1.3 二阶导数的缺点

```
二阶导数的主要缺点：

  1. 对噪声极其敏感！
  ────────────────────

  为什么？
  求一次导数已经会放大噪声
  求两次导数，噪声被放大得更多！

  原始信号+噪声：
    │   ╱╲ ╱╲   ╱╲
    │  ╱  ╲╱  ╲ ╱  ╲
    │─╱        ╲    ╲
    └────────────────────→

  一阶导数：                二阶导数：
    │ ∧ ∧ ∧                  │  ∧∧∧∧∧
    │∧│∧│∧│                  │ ∧│││││∧
    │││││││∧                 │∧│││││││∧
    └────────→               └──────────→
    噪声被放大                噪声被放大更多！

  解决方案：在使用Laplacian之前，必须先进行滤波！

  2. 可能产生双边缘
  ─────────────────
  对于某些边缘（如屋顶边缘），会检测出两条边
  这有时是好事，有时是问题

  3. 没有方向信息
  ───────────────
  Laplacian只给出边缘强度，不告诉你边缘方向
  如果需要方向信息，还是要用Sobel/Scharr
```

---

### 2. Laplacian的数学原理

#### 2.1 二阶导数的定义

```
连续情况下的Laplacian算子：

  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  一维二阶导数：                                          │
  │                                                         │
  │    d²f    d  df                                         │
  │    ─── = ── (──)                                        │
  │    dx²   dx  dx                                         │
  │                                                         │
  │  二维Laplacian（拉普拉斯算子）：                         │
  │                                                         │
  │           ∂²f    ∂²f                                    │
  │    ∇²f = ─── + ────                                     │
  │           ∂x²   ∂y²                                     │
  │                                                         │
  │  ∇² 读作 "nabla squared" 或 "Laplacian"                 │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

  物理意义：
  Laplacian 描述的是一个点与其周围点的差异程度
  • 如果一个点比周围都亮 → Laplacian为负
  • 如果一个点比周围都暗 → Laplacian为正
  • 如果一个点与周围相同 → Laplacian为零
```

#### 2.2 离散近似

```
离散情况下的二阶导数：

  一维离散二阶导数：

  f''(x) ≈ f(x+1) - 2f(x) + f(x-1)

  推导过程：
  f'(x) ≈ f(x+1) - f(x)        （右差分）
  f'(x-1) ≈ f(x) - f(x-1)      （左差分）
  f''(x) = f'(x) - f'(x-1)
        = [f(x+1) - f(x)] - [f(x) - f(x-1)]
        = f(x+1) - 2f(x) + f(x-1)

  对应的卷积核：[1, -2, 1]


  二维离散Laplacian：

  ∇²f ≈ [f(x+1,y) - 2f(x,y) + f(x-1,y)]   ← x方向二阶导数
      + [f(x,y+1) - 2f(x,y) + f(x,y-1)]   ← y方向二阶导数

      = f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4f(x,y)
```

---

### 3. Laplacian卷积核

#### 3.1 标准Laplacian核

```
Laplacian的卷积核：

  4-邻域Laplacian核：             8-邻域Laplacian核：

  ┌────┬────┬────┐              ┌────┬────┬────┐
  │ 0  │ 1  │ 0  │              │ 1  │ 1  │ 1  │
  ├────┼────┼────┤              ├────┼────┼────┤
  │ 1  │ -4 │ 1  │              │ 1  │ -8 │ 1  │
  ├────┼────┼────┤              ├────┼────┼────┤
  │ 0  │ 1  │ 0  │              │ 1  │ 1  │ 1  │
  └────┴────┴────┘              └────┴────┴────┘

  4-邻域：只考虑上下左右4个邻居
  8-邻域：考虑所有8个邻居（包括对角）

  核的特点：
  • 中心系数为负（-4或-8）
  • 所有系数之和为0
  • 对均匀区域响应为0
  • 各向同性（不区分方向）

  也可以使用负版本（效果相反）：
  ┌────┬────┬────┐
  │ 0  │ -1 │ 0  │
  ├────┼────┼────┤
  │ -1 │ 4  │ -1 │
  ├────┼────┼────┤
  │ 0  │ -1 │ 0  │
  └────┴────┴────┘
```

#### 3.2 为什么系数和为0？

```
核系数和为0的意义：

  对于均匀区域（所有像素相同）：
  ┌────┬────┬────┐       ┌────┬────┬────┐
  │100 │100 │100 │       │ 0  │ 1  │ 0  │
  ├────┼────┼────┤   ⊗   ├────┼────┼────┤
  │100 │100 │100 │       │ 1  │ -4 │ 1  │
  ├────┼────┼────┤       ├────┼────┼────┤
  │100 │100 │100 │       │ 0  │ 1  │ 0  │
  └────┴────┴────┘       └────┴────┴────┘

  结果 = 0×100 + 1×100 + 0×100
       + 1×100 + (-4)×100 + 1×100
       + 0×100 + 1×100 + 0×100
       = 100 + 100 + 100 + 100 - 400
       = 0

  在没有变化的区域，Laplacian输出为0
  只有在有变化（边缘）的地方才有非零输出
  这正是我们想要的！
```

---

### 4. OpenCV中的Laplacian

#### 4.1 函数语法

```
cv2.Laplacian() 函数详解：

  dst = cv2.Laplacian(src, ddepth, ksize=1, scale=1, delta=0)

  参数说明：
  ┌────────────┬─────────────────────────────────────────────────┐
  │ 参数       │ 说明                                             │
  ├────────────┼─────────────────────────────────────────────────┤
  │ src        │ 输入图像（灰度图）                               │
  ├────────────┼─────────────────────────────────────────────────┤
  │ ddepth     │ 输出深度（推荐CV_64F或CV_16S）                   │
  │            │ 同样需要保留负值！                               │
  ├────────────┼─────────────────────────────────────────────────┤
  │ ksize      │ 核大小（1, 3, 5, 7等奇数）                       │
  │            │ ksize=1时使用3×3核: [[0,1,0],[1,-4,1],[0,1,0]]  │
  │            │ ksize=3时计算更精确                              │
  ├────────────┼─────────────────────────────────────────────────┤
  │ scale      │ 缩放因子（默认1）                                │
  ├────────────┼─────────────────────────────────────────────────┤
  │ delta      │ 偏移量（默认0）                                  │
  └────────────┴─────────────────────────────────────────────────┘

  使用示例：
  laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
  laplacian = np.abs(laplacian)
  laplacian = laplacian.astype(np.uint8)
```

#### 4.2 ksize参数的影响

```
不同ksize的效果：

  ksize=1（默认）：
  使用标准的3×3 Laplacian核
  ┌────┬────┬────┐
  │ 0  │ 1  │ 0  │
  ├────┼────┼────┤
  │ 1  │ -4 │ 1  │
  ├────┼────┼────┤
  │ 0  │ 1  │ 0  │
  └────┴────┴────┘
  对噪声非常敏感

  ksize=3：
  使用二阶Sobel核的组合计算
  实际上是: ∂²f/∂x² + ∂²f/∂y²
  使用Sobel核计算二阶导数
  稍微平滑，精度更高

  ksize=5,7：
  更大的核，更多的平滑
  对噪声更鲁棒，但边缘可能变粗

  选择建议：
  • 噪声少：ksize=1 或 3
  • 噪声多：ksize=5 或更大，或先进行滤波
  • 一般推荐：先高斯滤波，再用 ksize=3
```

---

### 5. Laplacian的特点与应用

#### 5.1 优缺点总结

```
Laplacian算子的优缺点：

  ✅ 优点：
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  1. 各向同性                                                 │
  │     • 一次运算检测所有方向的边缘                             │
  │     • 不需要合并多个方向的结果                               │
  │                                                             │
  │  2. 边缘定位精确                                             │
  │     • 过零点比峰值更容易定位                                 │
  │     • 适合亚像素精度边缘检测                                 │
  │                                                             │
  │  3. 可用于图像锐化                                           │
  │     • 锐化图像 = 原图 - Laplacian                           │
  │     • 增强边缘和细节                                         │
  │                                                             │
  │  4. 在LoG（高斯拉普拉斯）中作为基础                          │
  │     • Laplacian of Gaussian 是经典的边缘检测方法             │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  ❌ 缺点：
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  1. 对噪声极其敏感！                                         │
  │     • 二阶导数放大噪声                                       │
  │     • 必须配合预滤波使用                                     │
  │                                                             │
  │  2. 可能产生双边缘                                           │
  │     • 对屋顶型边缘会检测出两条                               │
  │     • 可能需要后处理                                         │
  │                                                             │
  │  3. 没有方向信息                                             │
  │     • 只告诉你哪里是边缘                                     │
  │     • 不告诉你边缘的方向                                     │
  │                                                             │
  │  4. 边缘可能不连续                                           │
  │     • 相比Canny，边缘连续性较差                              │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

#### 5.2 典型应用场景

```
Laplacian的应用场景：

  1. 图像锐化
  ──────────
  锐化后 = 原图 - α × Laplacian

  原图                   Laplacian              锐化后
  ┌──────────┐          ┌──────────┐          ┌──────────┐
  │   模糊   │    →     │  边缘    │    →     │  清晰    │
  │  的图像  │          │  突出    │          │  的图像  │
  └──────────┘          └──────────┘          └──────────┘

  2. LoG（Laplacian of Gaussian）
  ────────────────────────────────
  先高斯平滑，再Laplacian
  这是经典的边缘检测组合

  3. 过零点检测（Zero-Crossing）
  ────────────────────────────────
  通过检测Laplacian的过零点来精确定位边缘
  比检测峰值更准确

  4. Blob检测（斑点检测）
  ──────────────────────────
  多尺度LoG可用于检测图像中的斑点
  SIFT特征检测就用了这个原理
```

---

## 💻 代码实战

### 代码1：Laplacian基本用法

```python
"""
Laplacian算子的基本用法
学习如何使用cv2.Laplacian()进行边缘检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 100

    # 矩形
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)

    # 圆形
    cv2.circle(img, (280, 100), 50, 200, -1)

    # 三角形
    pts = np.array([[80, 200], [30, 280], [130, 280]], np.int32)
    cv2.fillPoly(img, [pts], 180)

    # 渐变区域
    for i in range(100):
        img[200:280, 180+i] = 100 + int(i * 1.5)

    return img

img = create_test_image()

print("测试图像已创建")
print(f"  尺寸: {img.shape}")

# ===================== 应用Laplacian =====================

# 使用CV_64F保留负值
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=1)

print(f"\nLaplacian结果：")
print(f"  范围: [{laplacian.min():.1f}, {laplacian.max():.1f}]")
print(f"  包含负值说明检测到了从亮到暗的边缘")

# 取绝对值
laplacian_abs = np.abs(laplacian)
laplacian_abs = np.clip(laplacian_abs, 0, 255).astype(np.uint8)

# 使用convertScaleAbs
laplacian_cv = cv2.convertScaleAbs(laplacian)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

# Laplacian（带正负）
im = axes[0, 1].imshow(laplacian, cmap='RdBu', vmin=-200, vmax=200)
axes[0, 1].set_title('Laplacian（带正负）\n红=正，蓝=负', fontsize=12)
axes[0, 1].axis('off')
plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

# Laplacian绝对值
axes[1, 0].imshow(laplacian_abs, cmap='gray')
axes[1, 0].set_title('Laplacian（绝对值）', fontsize=12)
axes[1, 0].axis('off')

# 说明
axes[1, 1].axis('off')
info = """
Laplacian算子说明：

1. 函数调用：
   cv2.Laplacian(img, ddepth, ksize)

2. 参数说明：
   • ddepth: 使用CV_64F保留负值
   • ksize: 核大小（1,3,5,7）
   • ksize=1 使用标准3×3核

3. 标准Laplacian核：
   ┌────┬────┬────┐
   │ 0  │ 1  │ 0  │
   ├────┼────┼────┤
   │ 1  │ -4 │ 1  │
   ├────┼────┼────┤
   │ 0  │ 1  │ 0  │
   └────┴────┴────┘

4. 特点：
   • 一次检测所有方向边缘
   • 对噪声敏感
   • 边缘处有过零点
"""
axes[1, 1].text(0.1, 0.5, info, fontsize=10,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Laplacian边缘检测基础', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplacian_basic.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图像已保存为 'laplacian_basic.png'")
```

---

### 代码2：Laplacian对噪声的敏感性

```python
"""
演示Laplacian算子对噪声的敏感性
以及预处理（滤波）的重要性
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_clean_image():
    """创建干净的测试图像"""
    img = np.zeros((200, 300), dtype=np.uint8)
    img[:] = 80
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (220, 100), 40, 180, -1)
    return img

def add_noise(img, sigma=20):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float64) + noise, 0, 255)
    return noisy.astype(np.uint8)

# 创建图像
clean = create_clean_image()
noisy = add_noise(clean, sigma=25)

print("图像已创建")
print(f"  干净图像")
print(f"  噪声图像（sigma=25）")

# ===================== 直接应用Laplacian =====================

# 对干净图像
lap_clean = cv2.Laplacian(clean, cv2.CV_64F, ksize=1)
lap_clean_abs = cv2.convertScaleAbs(lap_clean)

# 对噪声图像（不预处理）
lap_noisy_direct = cv2.Laplacian(noisy, cv2.CV_64F, ksize=1)
lap_noisy_direct_abs = cv2.convertScaleAbs(lap_noisy_direct)

# ===================== 使用预滤波 =====================

# 高斯滤波后再Laplacian
noisy_gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
lap_gaussian = cv2.Laplacian(noisy_gaussian, cv2.CV_64F, ksize=1)
lap_gaussian_abs = cv2.convertScaleAbs(lap_gaussian)

# 中值滤波后再Laplacian
noisy_median = cv2.medianBlur(noisy, 5)
lap_median = cv2.Laplacian(noisy_median, cv2.CV_64F, ksize=1)
lap_median_abs = cv2.convertScaleAbs(lap_median)

# 双边滤波后再Laplacian
noisy_bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
lap_bilateral = cv2.Laplacian(noisy_bilateral, cv2.CV_64F, ksize=1)
lap_bilateral_abs = cv2.convertScaleAbs(lap_bilateral)

# ===================== 可视化 =====================

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# 第一行：原始图像
axes[0, 0].imshow(clean, cmap='gray')
axes[0, 0].set_title('干净图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(lap_clean_abs, cmap='gray')
axes[0, 1].set_title('干净图像的Laplacian\n（理想结果）', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(noisy, cmap='gray')
axes[0, 2].set_title('噪声图像', fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(lap_noisy_direct_abs, cmap='gray')
axes[0, 3].set_title('❌ 直接Laplacian\n（噪声严重！）', fontsize=11, color='red')
axes[0, 3].axis('off')

# 第二行：滤波后的图像
axes[1, 0].imshow(noisy_gaussian, cmap='gray')
axes[1, 0].set_title('高斯滤波后', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(lap_gaussian_abs, cmap='gray')
axes[1, 1].set_title('高斯滤波+Laplacian', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(noisy_median, cmap='gray')
axes[1, 2].set_title('中值滤波后', fontsize=11)
axes[1, 2].axis('off')

axes[1, 3].imshow(lap_median_abs, cmap='gray')
axes[1, 3].set_title('中值滤波+Laplacian', fontsize=11)
axes[1, 3].axis('off')

# 第三行：双边滤波和总结
axes[2, 0].imshow(noisy_bilateral, cmap='gray')
axes[2, 0].set_title('双边滤波后', fontsize=11)
axes[2, 0].axis('off')

axes[2, 1].imshow(lap_bilateral_abs, cmap='gray')
axes[2, 1].set_title('✓ 双边滤波+Laplacian\n（效果最好）', fontsize=11, color='green')
axes[2, 1].axis('off')

# 总结
axes[2, 2].axis('off')
axes[2, 3].axis('off')

summary = """
Laplacian对噪声敏感的原因：

二阶导数会放大噪声！
f'' = f(x+1) - 2f(x) + f(x-1)
噪声的二阶差分会产生很大的值

解决方案：
1. 先滤波，再Laplacian
2. 使用更大的ksize
3. 使用LoG（高斯拉普拉斯）

滤波方法推荐：
• 高斯滤波：通用，稍微模糊边缘
• 中值滤波：对椒盐噪声效果好
• 双边滤波：保边效果最好（推荐）
"""
axes[2, 2].text(0, 0.5, summary, fontsize=10,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.suptitle('Laplacian对噪声的敏感性', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplacian_noise.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n对比图已保存为 'laplacian_noise.png'")
```

---

### 代码3：不同ksize的效果对比

```python
"""
对比Laplacian不同ksize参数的效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建带噪声的图像 =====================

def create_image_with_noise():
    """创建带轻微噪声的测试图像"""
    img = np.zeros((250, 350), dtype=np.uint8)
    img[:] = 80

    cv2.rectangle(img, (40, 40), (140, 140), 200, -1)
    cv2.circle(img, (240, 90), 50, 180, -1)
    cv2.line(img, (40, 180), (150, 230), 200, 5)

    # 添加轻微噪声
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

img = create_image_with_noise()

print("测试图像已创建（带轻微噪声）")

# ===================== 不同ksize的Laplacian =====================

ksize_list = [1, 3, 5, 7]
results = {}

for ksize in ksize_list:
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    lap_abs = cv2.convertScaleAbs(lap)
    results[ksize] = lap_abs

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像（带轻微噪声）', fontsize=11)
axes[0, 0].axis('off')

# ksize=1
axes[0, 1].imshow(results[1], cmap='gray')
axes[0, 1].set_title('ksize=1\n（标准3×3核，噪声明显）', fontsize=11)
axes[0, 1].axis('off')

# ksize=3
axes[0, 2].imshow(results[3], cmap='gray')
axes[0, 2].set_title('ksize=3\n（稍有改善）', fontsize=11)
axes[0, 2].axis('off')

# ksize=5
axes[1, 0].imshow(results[5], cmap='gray')
axes[1, 0].set_title('ksize=5\n（更多平滑）', fontsize=11)
axes[1, 0].axis('off')

# ksize=7
axes[1, 1].imshow(results[7], cmap='gray')
axes[1, 1].set_title('ksize=7\n（平滑最多，边缘变粗）', fontsize=11)
axes[1, 1].axis('off')

# 说明
axes[1, 2].axis('off')
info = """
ksize选择指南：

ksize=1:
• 使用标准3×3 Laplacian核
• 对噪声非常敏感
• 边缘最细

ksize=3:
• 使用Sobel核计算二阶导数
• 稍有平滑效果

ksize=5,7:
• 更大的核，更多平滑
• 抗噪声能力更强
• 但边缘会变粗

推荐做法：
不要依赖大ksize来抗噪
而是：先滤波 + ksize=1或3
"""
axes[1, 2].text(0.1, 0.5, info, fontsize=10,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Laplacian不同ksize对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplacian_ksize.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n对比图已保存为 'laplacian_ksize.png'")
```

---

### 代码4：Laplacian与Sobel对比

```python
"""
对比Laplacian和Sobel边缘检测的效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_scene():
    """创建测试场景"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 80

    # 各种形状
    cv2.rectangle(img, (30, 30), (120, 120), 200, -1)
    cv2.circle(img, (200, 80), 45, 180, -1)

    # 线条（会产生双边缘）
    cv2.line(img, (280, 30), (380, 130), 200, 6)

    # 渐变边缘
    for i in range(80):
        img[160:260, 30+i] = 80 + int(i * 1.5)

    # 屋顶型边缘（细线条）
    cv2.line(img, (150, 200), (250, 200), 200, 2)

    # 复杂形状
    pts = np.array([[300, 160], [260, 280], [340, 280]], np.int32)
    cv2.fillPoly(img, [pts], 200)

    return img

img = create_test_scene()

# 预处理
img_blur = cv2.GaussianBlur(img, (3, 3), 0)

print("测试图像已创建")

# ===================== Sobel边缘检测 =====================

sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)

# Sobel梯度方向
sobel_dir = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

# ===================== Laplacian边缘检测 =====================

laplacian = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3)
laplacian_abs = cv2.convertScaleAbs(laplacian)

# ===================== 过零点检测 =====================

def find_zero_crossings(laplacian_img, threshold=10):
    """检测Laplacian的过零点"""
    # 过零点：相邻像素符号相反
    zero_crossings = np.zeros_like(laplacian_img, dtype=np.uint8)

    for i in range(1, laplacian_img.shape[0]-1):
        for j in range(1, laplacian_img.shape[1]-1):
            neighbors = [
                laplacian_img[i-1, j], laplacian_img[i+1, j],
                laplacian_img[i, j-1], laplacian_img[i, j+1]
            ]
            current = laplacian_img[i, j]

            # 检查是否有符号变化
            for n in neighbors:
                if current * n < 0 and abs(current - n) > threshold:
                    zero_crossings[i, j] = 255
                    break

    return zero_crossings

zero_cross = find_zero_crossings(laplacian, threshold=20)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

# Sobel
axes[0, 1].imshow(sobel_mag, cmap='gray')
axes[0, 1].set_title('Sobel（一阶导数）\n检测梯度大的区域', fontsize=12)
axes[0, 1].axis('off')

# Laplacian
axes[0, 2].imshow(laplacian_abs, cmap='gray')
axes[0, 2].set_title('Laplacian（二阶导数）\n检测二阶变化', fontsize=12)
axes[0, 2].axis('off')

# Sobel方向
mask = sobel_mag > 30
sobel_dir_masked = np.where(mask, sobel_dir, np.nan)
im = axes[1, 0].imshow(sobel_dir_masked, cmap='hsv', vmin=-180, vmax=180)
axes[1, 0].set_title('Sobel梯度方向\n（有方向信息）', fontsize=12)
axes[1, 0].axis('off')
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# Laplacian带正负
im2 = axes[1, 1].imshow(laplacian, cmap='RdBu', vmin=-100, vmax=100)
axes[1, 1].set_title('Laplacian（带正负）\n红=正，蓝=负', fontsize=12)
axes[1, 1].axis('off')
plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

# 过零点
axes[1, 2].imshow(zero_cross, cmap='gray')
axes[1, 2].set_title('Laplacian过零点\n（精确边缘位置）', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle('Sobel vs Laplacian 边缘检测对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_vs_laplacian.png', dpi=150, bbox_inches='tight')
plt.show()

# 打印对比总结
print("\nSobel vs Laplacian 对比：")
print("=" * 50)
print(f"{'特性':<20} | {'Sobel':<12} | {'Laplacian':<12}")
print("-" * 50)
print(f"{'导数阶数':<20} | {'一阶':<12} | {'二阶':<12}")
print(f"{'方向信息':<20} | {'有':<12} | {'无':<12}")
print(f"{'边缘定位':<20} | {'峰值':<12} | {'过零点':<12}")
print(f"{'噪声敏感度':<20} | {'中等':<12} | {'高':<12}")
print(f"{'双边缘问题':<20} | {'无':<12} | {'有':<12}")
print("=" * 50)

print("\n图像已保存为 'sobel_vs_laplacian.png'")
```

---

### 代码5：LoG（Laplacian of Gaussian）

```python
"""
LoG（Laplacian of Gaussian）边缘检测
高斯滤波+Laplacian的组合方法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_noisy_image():
    """创建带噪声的测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 80

    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (280, 100), 50, 180, -1)
    cv2.ellipse(img, (100, 220), (60, 40), 0, 0, 360, 200, -1)

    # 添加噪声
    noise = np.random.normal(0, 15, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

img = create_noisy_image()

print("带噪声的测试图像已创建")

# ===================== 不同方法对比 =====================

# 方法1：直接Laplacian
lap_direct = cv2.Laplacian(img, cv2.CV_64F, ksize=1)
lap_direct_abs = cv2.convertScaleAbs(lap_direct)

# 方法2：先高斯滤波，再Laplacian（LoG）
def apply_log(img, sigma=1.0):
    """应用LoG（Laplacian of Gaussian）"""
    # 根据sigma计算核大小
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    # 高斯滤波
    gaussian = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    # Laplacian
    laplacian = cv2.Laplacian(gaussian, cv2.CV_64F, ksize=1)

    return laplacian, gaussian

# 不同sigma的LoG
sigmas = [1.0, 2.0, 3.0]
log_results = {}

for sigma in sigmas:
    lap, gauss = apply_log(img, sigma)
    log_results[sigma] = {
        'gaussian': gauss,
        'laplacian': cv2.convertScaleAbs(lap),
        'raw': lap
    }

# 方法3：使用OpenCV的Laplacian with large ksize
lap_ksize5 = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
lap_ksize5_abs = cv2.convertScaleAbs(lap_ksize5)

# ===================== 可视化 =====================

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# 第一行：原图和直接Laplacian
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始噪声图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(lap_direct_abs, cmap='gray')
axes[0, 1].set_title('❌ 直接Laplacian\n（噪声严重）', fontsize=11, color='red')
axes[0, 1].axis('off')

axes[0, 2].imshow(lap_ksize5_abs, cmap='gray')
axes[0, 2].set_title('Laplacian (ksize=5)\n（稍有改善）', fontsize=11)
axes[0, 2].axis('off')

# 说明
axes[0, 3].axis('off')
info = """
LoG = Laplacian of Gaussian

原理：
1. 先用高斯滤波平滑噪声
2. 再用Laplacian检测边缘

LoG的优点：
• 抑制噪声
• 保留边缘

sigma的影响：
• sigma小：保留更多细节
• sigma大：更多平滑，边缘更粗
"""
axes[0, 3].text(0.1, 0.5, info, fontsize=10,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# 第二行：不同sigma的高斯滤波结果
for i, sigma in enumerate(sigmas):
    axes[1, i].imshow(log_results[sigma]['gaussian'], cmap='gray')
    axes[1, i].set_title(f'高斯滤波 (σ={sigma})', fontsize=11)
    axes[1, i].axis('off')

axes[1, 3].axis('off')
axes[1, 3].text(0.5, 0.5, 'σ越大\n图像越模糊', fontsize=12,
                ha='center', va='center', fontfamily='SimHei')

# 第三行：不同sigma的LoG结果
for i, sigma in enumerate(sigmas):
    axes[2, i].imshow(log_results[sigma]['laplacian'], cmap='gray')
    axes[2, i].set_title(f'✓ LoG (σ={sigma})', fontsize=11, color='green')
    axes[2, i].axis('off')

axes[2, 3].axis('off')
axes[2, 3].text(0.5, 0.5, 'σ越大\n边缘越粗\n噪声越少', fontsize=12,
                ha='center', va='center', fontfamily='SimHei')

plt.suptitle('LoG（Laplacian of Gaussian）边缘检测', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplacian_of_gaussian.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLoG对比图已保存为 'laplacian_of_gaussian.png'")
```

---

### 代码6：Laplacian用于图像锐化

```python
"""
使用Laplacian进行图像锐化
锐化 = 原图 - α × Laplacian
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建或读取图像 =====================

def create_blur_image():
    """创建一个稍微模糊的测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 绘制场景
    img[:] = 120

    # 建筑
    cv2.rectangle(img, (50, 100), (150, 250), 80, -1)
    cv2.rectangle(img, (70, 120), (90, 160), 180, -1)  # 窗户
    cv2.rectangle(img, (110, 120), (130, 160), 180, -1)  # 窗户
    cv2.rectangle(img, (90, 190), (120, 250), 60, -1)  # 门

    # 树
    cv2.rectangle(img, (220, 180), (240, 250), 100, -1)  # 树干
    cv2.circle(img, (230, 140), 50, 70, -1)  # 树冠

    # 太阳
    cv2.circle(img, (330, 60), 30, 200, -1)

    # 添加轻微模糊
    img = cv2.GaussianBlur(img, (5, 5), 1)

    return img

img = create_blur_image()

print("模糊图像已创建")

# ===================== 图像锐化 =====================

def laplacian_sharpening(img, alpha=1.0):
    """
    使用Laplacian进行图像锐化
    锐化图像 = 原图 - alpha × Laplacian

    注意：这里用减法是因为我们使用的Laplacian核中心为负
    如果核中心为正，则用加法
    """
    # 计算Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)

    # 锐化：原图 - alpha × Laplacian
    sharpened = img.astype(np.float64) - alpha * laplacian

    # 裁剪到有效范围
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened, laplacian

# 不同alpha值的锐化效果
alphas = [0.5, 1.0, 1.5, 2.0]
sharpened_results = {}

for alpha in alphas:
    sharp, lap = laplacian_sharpening(img, alpha)
    sharpened_results[alpha] = sharp

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像（稍模糊）', fontsize=12)
axes[0, 0].axis('off')

# Laplacian
_, laplacian = laplacian_sharpening(img, 1.0)
axes[0, 1].imshow(cv2.convertScaleAbs(laplacian), cmap='gray')
axes[0, 1].set_title('Laplacian（边缘信息）', fontsize=12)
axes[0, 1].axis('off')

# alpha=1.0的锐化结果
axes[0, 2].imshow(sharpened_results[1.0], cmap='gray')
axes[0, 2].set_title('锐化结果 (α=1.0)', fontsize=12)
axes[0, 2].axis('off')

# 不同alpha对比
axes[1, 0].imshow(sharpened_results[0.5], cmap='gray')
axes[1, 0].set_title('α=0.5（轻微锐化）', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(sharpened_results[1.5], cmap='gray')
axes[1, 1].set_title('α=1.5（较强锐化）', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(sharpened_results[2.0], cmap='gray')
axes[1, 2].set_title('α=2.0（过度锐化）', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle('Laplacian图像锐化', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplacian_sharpening.png', dpi=150, bbox_inches='tight')
plt.show()

# 绘制一维剖面对比
fig, ax = plt.subplots(figsize=(12, 5))

row = 140
ax.plot(img[row, :], 'b-', linewidth=2, label='原图')
ax.plot(sharpened_results[1.0][row, :], 'r-', linewidth=2, label='锐化后 (α=1.0)')
ax.set_xlabel('像素位置', fontsize=12)
ax.set_ylabel('灰度值', fontsize=12)
ax.set_title(f'第{row}行的灰度值对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('laplacian_sharpening_profile.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n锐化效果图已保存")
print("\n锐化原理：")
print("  锐化图像 = 原图 - α × Laplacian")
print("  • α < 1: 轻微锐化")
print("  • α = 1: 标准锐化")
print("  • α > 1: 强烈锐化（可能过度）")
```

---

## 📝 本节总结

```
┌────────────────────────────────────────────────────────────────────┐
│                       Laplacian算子总结                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. Laplacian是什么？                                               │
│     • 二阶导数边缘检测算子                                         │
│     • 公式：∇²f = ∂²f/∂x² + ∂²f/∂y²                                │
│     • 一次运算检测所有方向边缘                                     │
│                                                                    │
│  2. Laplacian卷积核                                                 │
│                                                                    │
│     4-邻域核：                   8-邻域核：                        │
│     ┌────┬────┬────┐           ┌────┬────┬────┐                   │
│     │ 0  │ 1  │ 0  │           │ 1  │ 1  │ 1  │                   │
│     │ 1  │ -4 │ 1  │           │ 1  │ -8 │ 1  │                   │
│     │ 0  │ 1  │ 0  │           │ 1  │ 1  │ 1  │                   │
│     └────┴────┴────┘           └────┴────┴────┘                   │
│                                                                    │
│  3. 与一阶导数的区别                                                │
│     ┌────────────────┬────────────────┬────────────────┐          │
│     │                │  一阶（Sobel） │ 二阶（Laplacian）│          │
│     ├────────────────┼────────────────┼────────────────┤          │
│     │  边缘位置      │    峰值处      │    过零点处    │          │
│     ├────────────────┼────────────────┼────────────────┤          │
│     │  方向信息      │      有        │      无        │          │
│     ├────────────────┼────────────────┼────────────────┤          │
│     │  噪声敏感度    │     中等       │      高        │          │
│     └────────────────┴────────────────┴────────────────┘          │
│                                                                    │
│  4. OpenCV函数                                                      │
│     cv2.Laplacian(src, ddepth, ksize=1)                            │
│                                                                    │
│  5. 重要注意事项                                                    │
│     • 对噪声极其敏感，必须先滤波！                                 │
│     • 使用CV_64F保留负值                                           │
│     • 可能产生双边缘                                               │
│                                                                    │
│  6. 主要应用                                                        │
│     • LoG边缘检测（高斯+Laplacian）                                │
│     • 图像锐化（原图 - α×Laplacian）                               │
│     • 过零点检测                                                   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 下一步学习

在下一节 **"05-Canny边缘检测"** 中，我们将：
- 学习公认最优的边缘检测算法
- 了解Canny的多阶段处理流程
- 掌握双阈值和边缘连接的原理

---

> 💡 **学习建议**：Laplacian虽然对噪声敏感，但理解二阶导数对于学习更高级的图像处理算法很重要。记住：使用Laplacian前一定要先滤波！
