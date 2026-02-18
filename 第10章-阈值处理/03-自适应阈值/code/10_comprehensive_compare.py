"""
全局阈值与自适应阈值的综合对比
在复杂光照条件下比较固定阈值、Otsu、自适应均值和自适应高斯
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 综合对比函数 =====================

def comprehensive_comparison(image):
    """全面比较不同阈值方法"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    results = {}

    # 1. 固定阈值
    _, results['固定阈值 (T=127)'] = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 2. Otsu
    ret, results['Otsu'] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 自适应均值
    results['自适应均值'] = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    # 4. 自适应高斯
    results['自适应高斯'] = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)

    return results

# ===================== 创建具有明显光照不均的测试图像 =====================

def create_challenging_image():
    img = np.ones((400, 500), dtype=np.uint8) * 200

    lines = [
        "Comparison Test",
        "Global vs Adaptive",
        "Which is better?",
        "Let's find out!",
    ]

    for i, line in enumerate(lines):
        y = 80 + i * 80
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 30, 2)

    # 复杂光照变化
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            factor = 1.0 - 0.3 * (i / rows) - 0.3 * (j / cols)
            factor += 0.1 * np.sin(i / 30) * np.cos(j / 40)
            img[i, j] = int(img[i, j] * max(0.3, min(1.0, factor)))

    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# ===================== 运行对比 =====================

test_img = create_challenging_image()
results = comprehensive_comparison(test_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 原图
axes[0].imshow(test_img, cmap='gray')
axes[0].set_title('原始图像\n(复杂光照)', fontsize=11)
axes[0].axis('off')

# 各种方法的结果
for i, (name, result) in enumerate(results.items(), 1):
    axes[i].imshow(result, cmap='gray')
    axes[i].set_title(name, fontsize=11)
    axes[i].axis('off')

axes[5].axis('off')

plt.suptitle('全局阈值 vs 自适应阈值 综合对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('comprehensive_compare.png', dpi=150, bbox_inches='tight')
plt.show()

# 打印比较结论
print("方法对比结论：")
print("=" * 60)
print("固定阈值:       简单快速，但无法处理光照不均")
print("Otsu:          自动选择阈值，但仍是全局阈值")
print("自适应均值:     局部阈值，对边缘敏感")
print("自适应高斯:     局部阈值，边缘更平滑（推荐）")
print("=" * 60)
