# Otsu阈值

## 1. Otsu算法简介

### 1.1 什么是Otsu阈值？

Otsu阈值法（大津算法）是一种**自动确定最佳阈值**的方法，由日本学者大津展之（Nobuyuki Otsu）于1979年提出。

> **核心思想**：寻找一个阈值，使得分割后的前景和背景两类像素的**类间方差最大**。

简单来说，Otsu算法会自动帮你找到一个最佳阈值，让前景和背景"分得最开"。

```
                    直方图示意
频率 │
     │    ****            ****
     │   *    *          *    *
     │  *      *        *      *
     │ *        *      *        *
     │*          *    *          *
     │            ****
     └──────────────┼───────────────── 像素值
     0            最佳阈值T           255
                   ↑
              Otsu自动找到
```

### 1.2 为什么需要自动阈值？

| 场景 | 手动选择阈值的问题 |
|------|-------------------|
| **批量处理** | 每张图像的最佳阈值可能不同 |
| **实时处理** | 没时间手动调整 |
| **未知图像** | 事先不知道图像特征 |
| **动态变化** | 图像亮度随时间变化 |

Otsu算法可以自动适应不同图像的特点，无需人工干预。

### 1.3 适用场景

Otsu算法最适合的图像是具有**双峰直方图**的图像：

```python
"""
示例1：理想的双峰直方图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_bimodal_image():
    """创建具有明显双峰分布的图像"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 左半部分：暗色（背景）
    img[:, :200] = np.random.normal(60, 15, (300, 200)).clip(0, 255)

    # 右半部分：亮色（前景）
    img[:, 200:] = np.random.normal(190, 15, (300, 200)).clip(0, 255)

    return img.astype(np.uint8)

# 创建测试图像
bimodal_img = create_bimodal_image()

# 计算直方图
hist = cv2.calcHist([bimodal_img], [0], None, [256], [0, 256]).flatten()

# 应用Otsu阈值
otsu_thresh, binary = cv2.threshold(bimodal_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(bimodal_img, cmap='gray')
axes[0].set_title('Original Image\n(Bimodal distribution)', fontsize=11)
axes[0].axis('off')

axes[1].plot(hist)
axes[1].axvline(x=otsu_thresh, color='r', linestyle='--', linewidth=2, label=f'Otsu T={otsu_thresh:.0f}')
axes[1].fill_between(range(256), hist, alpha=0.3)
axes[1].set_title('Histogram\n(Two clear peaks)', fontsize=11)
axes[1].set_xlabel('Pixel Value')
axes[1].set_ylabel('Frequency')
axes[1].legend()

axes[2].imshow(binary, cmap='gray')
axes[2].set_title(f'Otsu Result\n(T={otsu_thresh:.0f})', fontsize=11)
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(f"Otsu自动选择的阈值: {otsu_thresh:.0f}")
print("此阈值正好位于两个峰之间的谷底位置")
```

---

## 2. Otsu算法原理

### 2.1 数学基础

Otsu算法的目标是找到阈值T，使得**类间方差**σ²_B最大。

#### 基本概念

假设图像有L个灰度级（0到L-1），像素总数为N：

```
p(i) = n(i) / N    # 灰度级i的概率（归一化直方图）
n(i)               # 灰度级i的像素数
```

#### 类别划分

用阈值T将像素分为两类：
- **类0（背景）**：灰度值 ≤ T
- **类1（前景）**：灰度值 > T

```
各类的概率：
ω₀(T) = Σ p(i)     (i = 0 到 T)      # 背景概率
ω₁(T) = Σ p(i)     (i = T+1 到 L-1)   # 前景概率

各类的均值：
μ₀(T) = Σ i·p(i) / ω₀(T)    (i = 0 到 T)
μ₁(T) = Σ i·p(i) / ω₁(T)    (i = T+1 到 L-1)

全局均值：
μ = Σ i·p(i)    (i = 0 到 L-1)
```

#### 类间方差

```
σ²_B(T) = ω₀(T) · ω₁(T) · (μ₀(T) - μ₁(T))²
```

**Otsu算法就是找使得σ²_B最大的T值。**

### 2.2 算法可视化

```python
"""
示例2：可视化Otsu算法的工作原理
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_otsu_principle(image):
    """可视化Otsu算法原理"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 计算归一化直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()

    # 计算每个阈值的类间方差
    between_class_variance = np.zeros(256)

    for T in range(256):
        # 类0和类1的概率
        w0 = hist_norm[:T+1].sum()
        w1 = hist_norm[T+1:].sum()

        if w0 == 0 or w1 == 0:
            continue

        # 类0和类1的均值
        indices = np.arange(256)
        mu0 = (indices[:T+1] * hist_norm[:T+1]).sum() / w0
        mu1 = (indices[T+1:] * hist_norm[T+1:]).sum() / w1

        # 类间方差
        between_class_variance[T] = w0 * w1 * (mu0 - mu1) ** 2

    # 找到最大类间方差对应的阈值
    optimal_T = np.argmax(between_class_variance)

    # OpenCV的Otsu结果（验证）
    cv_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 原图
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')

    # 直方图
    axes[0, 1].bar(range(256), hist_norm, width=1, alpha=0.7)
    axes[0, 1].axvline(x=optimal_T, color='r', linestyle='--', linewidth=2,
                       label=f'Optimal T={optimal_T}')
    axes[0, 1].set_title('Normalized Histogram', fontsize=12)
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].legend()

    # 类间方差曲线
    axes[1, 0].plot(between_class_variance, 'b-', linewidth=1.5)
    axes[1, 0].axvline(x=optimal_T, color='r', linestyle='--', linewidth=2)
    axes[1, 0].scatter([optimal_T], [between_class_variance[optimal_T]],
                       color='r', s=100, zorder=5, label=f'Max at T={optimal_T}')
    axes[1, 0].set_title('Between-Class Variance σ²_B(T)', fontsize=12)
    axes[1, 0].set_xlabel('Threshold T')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 结果
    _, binary = cv2.threshold(gray, optimal_T, 255, cv2.THRESH_BINARY)
    axes[1, 1].imshow(binary, cmap='gray')
    axes[1, 1].set_title(f'Otsu Result (T={optimal_T})', fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"手动计算的Otsu阈值: {optimal_T}")
    print(f"OpenCV计算的阈值: {cv_thresh:.0f}")
    print(f"最大类间方差: {between_class_variance[optimal_T]:.4f}")

    return optimal_T

# 创建测试图像
test_img = create_bimodal_image()  # 使用之前定义的函数
optimal_threshold = visualize_otsu_principle(test_img)
```

### 2.3 手动实现Otsu

```python
"""
示例3：手动实现Otsu算法
"""
import cv2
import numpy as np

def otsu_threshold_manual(image):
    """
    手动实现Otsu阈值算法

    Parameters:
    -----------
    image : numpy.ndarray
        输入灰度图像

    Returns:
    --------
    optimal_threshold : int
        最佳阈值
    binary : numpy.ndarray
        二值化结果
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 计算直方图
    hist = np.zeros(256)
    for pixel in gray.ravel():
        hist[pixel] += 1

    # 归一化
    total_pixels = gray.size
    hist_norm = hist / total_pixels

    # 全局均值
    global_mean = np.sum(np.arange(256) * hist_norm)

    # 遍历所有可能的阈值
    max_variance = 0
    optimal_threshold = 0

    cumulative_sum = 0      # 累积概率
    cumulative_mean = 0     # 累积均值

    for T in range(256):
        # 更新累积值
        cumulative_sum += hist_norm[T]
        cumulative_mean += T * hist_norm[T]

        if cumulative_sum == 0 or cumulative_sum == 1:
            continue

        # 类0和类1的概率
        w0 = cumulative_sum
        w1 = 1 - cumulative_sum

        # 类0和类1的均值
        mu0 = cumulative_mean / w0
        mu1 = (global_mean - cumulative_mean) / w1

        # 类间方差
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > max_variance:
            max_variance = variance
            optimal_threshold = T

    # 应用阈值
    binary = np.where(gray > optimal_threshold, 255, 0).astype(np.uint8)

    return optimal_threshold, binary

# 测试
test_img = create_bimodal_image()

# 手动实现
manual_thresh, manual_binary = otsu_threshold_manual(test_img)

# OpenCV实现
cv_thresh, cv_binary = cv2.threshold(test_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"手动实现的Otsu阈值: {manual_thresh}")
print(f"OpenCV的Otsu阈值: {cv_thresh:.0f}")
print(f"两者是否一致: {manual_thresh == int(cv_thresh)}")
```

---

## 3. OpenCV中的Otsu阈值

### 3.1 基本使用

```python
"""
示例4：OpenCV中Otsu阈值的使用
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Otsu阈值的基本使用方式
def demo_otsu_usage():
    """演示Otsu阈值的基本使用"""

    # 创建测试图像
    img = np.zeros((200, 300), dtype=np.uint8)
    img[:, :150] = 50   # 左半部分暗
    img[:, 150:] = 200  # 右半部分亮

    # 添加噪声
    noise = np.random.normal(0, 20, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # 使用Otsu阈值
    # 注意：thresh参数设为0，因为Otsu会自动计算
    # 类型使用 THRESH_BINARY + THRESH_OTSU
    otsu_thresh, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print("=" * 50)
    print("Otsu阈值使用方法：")
    print("=" * 50)
    print("代码：")
    print("  ret, binary = cv2.threshold(gray, 0, 255,")
    print("                cv2.THRESH_BINARY + cv2.THRESH_OTSU)")
    print()
    print("参数说明：")
    print("  - thresh=0: 传入的阈值会被忽略")
    print("  - 返回的ret: 实际使用的Otsu阈值")
    print("  - THRESH_OTSU: 启用Otsu自动阈值")
    print()
    print(f"本例中Otsu自动选择的阈值: {otsu_thresh:.0f}")
    print("=" * 50)

    return img, binary, otsu_thresh

img, binary, thresh = demo_otsu_usage()

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original', fontsize=12)
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title(f'Otsu (T={thresh:.0f})', fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

### 3.2 Otsu与其他类型组合

```python
"""
示例5：Otsu与不同阈值类型的组合
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def otsu_with_different_types(image):
    """Otsu与不同阈值类型组合"""

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Otsu可以与BINARY或BINARY_INV组合
    thresh1, binary = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2, binary_inv = cv2.threshold(gray, 0, 255,
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 注意：Otsu不能与TRUNC、TOZERO等组合（会报错或效果不理想）

    return {
        'original': gray,
        'binary': (thresh1, binary),
        'binary_inv': (thresh2, binary_inv)
    }

# 创建测试图像
test_img = np.zeros((200, 300), dtype=np.uint8)
cv2.circle(test_img, (150, 100), 60, 200, -1)
cv2.rectangle(test_img, (20, 20), (80, 80), 180, -1)
noise = np.random.normal(0, 15, test_img.shape)
test_img = np.clip(test_img + noise, 0, 255).astype(np.uint8)

results = otsu_with_different_types(test_img)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(results['original'], cmap='gray')
axes[0].set_title('Original', fontsize=12)
axes[0].axis('off')

axes[1].imshow(results['binary'][1], cmap='gray')
axes[1].set_title(f"BINARY + OTSU\n(T={results['binary'][0]:.0f})", fontsize=11)
axes[1].axis('off')

axes[2].imshow(results['binary_inv'][1], cmap='gray')
axes[2].set_title(f"BINARY_INV + OTSU\n(T={results['binary_inv'][0]:.0f})", fontsize=11)
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("Otsu阈值组合使用：")
print("- cv2.THRESH_BINARY + cv2.THRESH_OTSU: 标准Otsu")
print("- cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU: 反向Otsu")
print(f"\n两种方式使用相同的阈值: {results['binary'][0]:.0f}")
```

### 3.3 Triangle阈值（补充）

```python
"""
示例6：Triangle阈值 - 另一种自动阈值方法
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_otsu_triangle(image):
    """比较Otsu和Triangle阈值"""

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Otsu阈值
    otsu_thresh, otsu_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Triangle阈值
    triangle_thresh, triangle_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    return {
        'original': gray,
        'otsu': (otsu_thresh, otsu_binary),
        'triangle': (triangle_thresh, triangle_binary)
    }

# 创建单峰分布的图像（更适合Triangle）
def create_unimodal_image():
    """创建单峰分布的图像"""
    img = np.random.normal(180, 30, (300, 400)).clip(0, 255).astype(np.uint8)

    # 添加少量暗色目标
    cv2.circle(img, (100, 150), 40, 30, -1)
    cv2.rectangle(img, (250, 100), (350, 200), 50, -1)

    return img

# 测试两种图像
bimodal_img = create_bimodal_image()
unimodal_img = create_unimodal_image()

# 双峰图像
bimodal_results = compare_otsu_triangle(bimodal_img)

# 单峰图像
unimodal_results = compare_otsu_triangle(unimodal_img)

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 双峰图像
axes[0, 0].imshow(bimodal_results['original'], cmap='gray')
axes[0, 0].set_title('Bimodal Image', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].hist(bimodal_results['original'].ravel(), 256, [0, 256])
axes[0, 1].set_title('Histogram (Two peaks)', fontsize=11)
axes[0, 1].axvline(x=bimodal_results['otsu'][0], color='r', linestyle='--', label=f"Otsu={bimodal_results['otsu'][0]:.0f}")
axes[0, 1].axvline(x=bimodal_results['triangle'][0], color='g', linestyle=':', label=f"Triangle={bimodal_results['triangle'][0]:.0f}")
axes[0, 1].legend(fontsize=8)

axes[0, 2].imshow(bimodal_results['otsu'][1], cmap='gray')
axes[0, 2].set_title(f"Otsu (T={bimodal_results['otsu'][0]:.0f})", fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(bimodal_results['triangle'][1], cmap='gray')
axes[0, 3].set_title(f"Triangle (T={bimodal_results['triangle'][0]:.0f})", fontsize=11)
axes[0, 3].axis('off')

# 单峰图像
axes[1, 0].imshow(unimodal_results['original'], cmap='gray')
axes[1, 0].set_title('Unimodal Image', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].hist(unimodal_results['original'].ravel(), 256, [0, 256])
axes[1, 1].set_title('Histogram (One peak)', fontsize=11)
axes[1, 1].axvline(x=unimodal_results['otsu'][0], color='r', linestyle='--', label=f"Otsu={unimodal_results['otsu'][0]:.0f}")
axes[1, 1].axvline(x=unimodal_results['triangle'][0], color='g', linestyle=':', label=f"Triangle={unimodal_results['triangle'][0]:.0f}")
axes[1, 1].legend(fontsize=8)

axes[1, 2].imshow(unimodal_results['otsu'][1], cmap='gray')
axes[1, 2].set_title(f"Otsu (T={unimodal_results['otsu'][0]:.0f})", fontsize=11)
axes[1, 2].axis('off')

axes[1, 3].imshow(unimodal_results['triangle'][1], cmap='gray')
axes[1, 3].set_title(f"Triangle (T={unimodal_results['triangle'][0]:.0f})", fontsize=11)
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()

print("Otsu vs Triangle 比较：")
print("=" * 50)
print("Otsu:     最适合双峰分布的图像")
print("Triangle: 最适合单峰分布的图像（少量目标）")
print("=" * 50)
```

---

## 4. 预处理与Otsu

### 4.1 高斯模糊预处理

```python
"""
示例7：高斯模糊预处理提升Otsu效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def otsu_with_preprocessing(image, blur_sizes=[0, 3, 5, 7]):
    """对比不同程度高斯模糊对Otsu的影响"""

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    results = []

    for ksize in blur_sizes:
        if ksize == 0:
            processed = gray.copy()
            name = "No blur"
        else:
            processed = cv2.GaussianBlur(gray, (ksize, ksize), 0)
            name = f"Blur {ksize}x{ksize}"

        thresh, binary = cv2.threshold(processed, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append({
            'name': name,
            'processed': processed,
            'binary': binary,
            'threshold': thresh
        })

    return results

# 创建带噪声的测试图像
def create_noisy_image():
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:, :200] = 70
    img[:, 200:] = 180

    cv2.circle(img, (100, 150), 50, 180, -1)
    cv2.rectangle(img, (250, 100), (350, 200), 70, -1)

    # 添加较强的噪声
    noise = np.random.normal(0, 25, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

noisy_img = create_noisy_image()
results = otsu_with_preprocessing(noisy_img)

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, res in enumerate(results):
    axes[0, i].imshow(res['processed'], cmap='gray')
    axes[0, i].set_title(res['name'], fontsize=11)
    axes[0, i].axis('off')

    axes[1, i].imshow(res['binary'], cmap='gray')
    axes[1, i].set_title(f"T={res['threshold']:.0f}", fontsize=11)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Preprocessed', fontsize=12)
axes[1, 0].set_ylabel('Otsu Result', fontsize=12)

plt.tight_layout()
plt.show()

print("高斯模糊预处理的效果：")
for res in results:
    print(f"  {res['name']:12s}: Otsu阈值 = {res['threshold']:.0f}")
print("\n推荐：对噪声图像先进行3x3或5x5的高斯模糊")
```

### 4.2 形态学预处理

```python
"""
示例8：形态学预处理配合Otsu
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphology_then_otsu(image):
    """形态学预处理后再Otsu"""

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 定义形态学核
    kernel = np.ones((5, 5), np.uint8)

    # 1. 直接Otsu
    _, direct_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. 形态学闭运算（填充小洞）后Otsu
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    _, closed_otsu = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 形态学开运算（去除噪点）后Otsu
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    _, opened_otsu = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 顶帽变换增强后Otsu
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.add(gray, tophat)
    _, tophat_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        'original': gray,
        'direct': direct_otsu,
        'closed': closed_otsu,
        'opened': opened_otsu,
        'tophat': tophat_otsu
    }

# 创建测试图像
test_img = create_noisy_image()
results = morphology_then_otsu(test_img)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

titles = ['Original', 'Direct Otsu', 'Close + Otsu',
          'Open + Otsu', 'TopHat + Otsu', '']
keys = ['original', 'direct', 'closed', 'opened', 'tophat', None]

for i, (title, key) in enumerate(zip(titles, keys)):
    if key is None:
        axes[i].axis('off')
    else:
        axes[i].imshow(results[key], cmap='gray')
        axes[i].set_title(title, fontsize=11)
        axes[i].axis('off')

plt.tight_layout()
plt.show()

print("形态学预处理的作用：")
print("- 闭运算: 填充前景中的小洞")
print("- 开运算: 去除小的噪点")
print("- 顶帽: 增强亮色细节")
```

---

## 5. 实际应用案例

### 5.1 医学图像分割

```python
"""
示例9：医学图像分割（模拟）
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_medical_image(image):
    """医学图像分割流程"""

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. 预处理：高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # 3. Otsu阈值分割
    thresh, binary = cv2.threshold(enhanced, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 形态学后处理
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. 提取轮廓
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制结果
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return {
        'gray': gray,
        'enhanced': enhanced,
        'binary': binary,
        'cleaned': cleaned,
        'result': result,
        'threshold': thresh,
        'contours': len(contours)
    }

# 创建模拟医学图像（如细胞图像）
def create_medical_image():
    img = np.random.normal(100, 20, (400, 500)).astype(np.uint8)

    # 添加"细胞"
    cells = [
        (100, 100, 30), (200, 150, 25), (350, 120, 35),
        (150, 280, 28), (300, 300, 32), (420, 250, 26)
    ]

    for x, y, r in cells:
        cv2.circle(img, (x, y), r, 200, -1)
        # 添加细胞内部结构
        cv2.circle(img, (x, y), r // 3, 230, -1)

    return img

medical_img = create_medical_image()
results = segment_medical_image(medical_img)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(results['gray'], cmap='gray')
axes[0, 0].set_title('Original Medical Image', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(results['enhanced'], cmap='gray')
axes[0, 1].set_title('CLAHE Enhanced', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(results['binary'], cmap='gray')
axes[0, 2].set_title(f"Otsu (T={results['threshold']})", fontsize=11)
axes[0, 2].axis('off')

axes[1, 0].imshow(results['cleaned'], cmap='gray')
axes[1, 0].set_title('Morphology Cleaned', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(results['result'], cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"Detected Objects: {results['contours']}", fontsize=11)
axes[1, 1].axis('off')

# 直方图
axes[1, 2].hist(results['gray'].ravel(), 256, [0, 256], alpha=0.5, label='Original')
axes[1, 2].hist(results['enhanced'].ravel(), 256, [0, 256], alpha=0.5, label='Enhanced')
axes[1, 2].axvline(x=results['threshold'], color='r', linestyle='--',
                   label=f"Otsu T={results['threshold']}")
axes[1, 2].set_title('Histograms', fontsize=11)
axes[1, 2].legend()

plt.tight_layout()
plt.show()

print(f"分割结果：检测到 {results['contours']} 个目标")
```

### 5.2 工业缺陷检测

```python
"""
示例10：工业缺陷检测
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_defects(image, min_area=50):
    """工业产品缺陷检测"""

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu阈值
    thresh, binary = cv2.threshold(blurred, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤小轮廓
    defects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            defects.append({
                'contour': contour,
                'area': area,
                'bbox': (x, y, w, h)
            })

    # 绘制结果
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for defect in defects:
        x, y, w, h = defect['bbox']
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(result, f"Area:{defect['area']:.0f}", (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return binary, result, defects

# 创建模拟产品图像（带缺陷）
def create_product_image():
    # 产品表面（均匀灰色）
    img = np.ones((400, 500), dtype=np.uint8) * 180

    # 添加一些正常纹理
    for i in range(400):
        img[i, :] += (5 * np.sin(i / 20)).astype(np.uint8)

    # 添加"缺陷"（暗色斑点、划痕等）
    cv2.circle(img, (100, 100), 20, 50, -1)  # 暗斑
    cv2.circle(img, (350, 150), 15, 60, -1)  # 暗斑
    cv2.line(img, (200, 50), (250, 150), 40, 3)  # 划痕
    cv2.ellipse(img, (400, 300), (25, 10), 30, 0, 360, 55, -1)  # 椭圆缺陷

    # 添加轻微噪声
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

product_img = create_product_image()
binary, result, defects = detect_defects(product_img)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(product_img, cmap='gray')
axes[0].set_title('Product Surface', fontsize=12)
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('Defect Mask (Otsu)', fontsize=12)
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'Detected Defects: {len(defects)}', fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(f"检测结果：发现 {len(defects)} 个缺陷")
for i, defect in enumerate(defects, 1):
    print(f"  缺陷 {i}: 面积 = {defect['area']:.0f} 像素")
```

---

## 6. 本节小结

### 6.1 核心要点

| 要点 | 说明 |
|------|------|
| **算法原理** | 最大化类间方差 |
| **适用场景** | 双峰分布的图像 |
| **使用方法** | `cv2.THRESH_BINARY + cv2.THRESH_OTSU` |
| **返回值** | 自动计算的最佳阈值 |

### 6.2 Otsu阈值速查

```python
# 标准用法
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 反向二值化
ret, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 带高斯模糊预处理
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
ret, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### 6.3 方法选择指南

```
选择阈值方法的决策树：
                    ┌─────────────────┐
                    │ 图像是否光照均匀？│
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │是                           │否
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │直方图是否双峰？  │           │ 使用自适应阈值    │
    └────────┬────────┘           └─────────────────┘
             │
   ┌─────────┴─────────┐
   │是                 │否
   ▼                   ▼
┌──────────┐     ┌──────────────┐
│ 使用Otsu │     │使用Triangle  │
└──────────┘     │ 或手动调整   │
                 └──────────────┘
```

---

## 7. 练习题

### 练习1：基础应用
对一张文档图像分别使用固定阈值、Otsu阈值和自适应阈值，对比效果差异。

### 练习2：算法实现
不使用OpenCV的Otsu功能，手动实现Otsu算法并验证结果。

### 练习3：实际项目
使用Otsu阈值实现一个简单的硬币计数程序，统计图像中的硬币数量。

---

## 8. 章节总结

恭喜你完成了第10章的学习！让我们回顾一下本章的重要内容：

### 阈值处理方法总结

| 方法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **固定阈值** | 已知最佳阈值 | 简单快速 | 需手动调整 |
| **Otsu** | 双峰分布图像 | 自动确定阈值 | 不适合光照不均 |
| **Triangle** | 单峰分布图像 | 自动确定阈值 | 应用场景有限 |
| **自适应阈值** | 光照不均匀 | 局部适应 | 参数需调整 |

### 下一步学习建议

1. **轮廓检测**：阈值处理后，通常需要进行轮廓检测
2. **形态学操作**：用于清理阈值化后的二值图像
3. **连通组件分析**：分析和计数二值图像中的目标

---

> **本章完成！** 你已经掌握了图像阈值处理的完整知识体系！
