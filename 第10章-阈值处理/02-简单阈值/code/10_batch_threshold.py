"""
批量阈值处理与对比
使用多个阈值批量处理图像并可视化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 批量处理函数 =====================

def batch_threshold(image, thresholds, thresh_type=cv2.THRESH_BINARY):
    """使用多个阈值批量处理图像"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    results = []
    for T in thresholds:
        _, binary = cv2.threshold(gray, T, 255, thresh_type)
        results.append((T, binary))

    return results

def visualize_batch_results(original, results, title="批量阈值处理结果"):
    """可视化批量处理结果"""
    n = len(results) + 1
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    # 原图
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('原始图像', fontsize=11)
    axes[0].axis('off')

    # 结果
    for i, (T, result) in enumerate(results, 1):
        if i < len(axes):
            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(f'T = {T}', fontsize=11)
            axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('batch_threshold.png', dpi=150, bbox_inches='tight')
    plt.show()

# ===================== 创建测试图像 =====================

img = np.zeros((200, 300), dtype=np.uint8)
cv2.circle(img, (150, 100), 80, 200, -1)
noise = np.random.normal(0, 30, img.shape)
img = np.clip(img + noise, 0, 255).astype(np.uint8)

# ===================== 批量处理 =====================

thresholds = [50, 80, 110, 140, 170, 200]
results = batch_threshold(img, thresholds)

# 可视化
visualize_batch_results(img, results, "不同阈值的效果对比")

print("批量阈值处理完成")
print(f"测试阈值: {thresholds}")
print("\n图像已保存为 'batch_threshold.png'")
