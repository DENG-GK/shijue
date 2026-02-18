"""
示例4：分段线性对比度增强
- 通过控制点定义分段映射函数
- 可灵活增强特定灰度范围
- 对比恒等、中间增强、S曲线、高对比度等变换
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def piecewise_linear_transform(image, points):
    """分段线性变换"""
    points = sorted(points, key=lambda x: x[0])
    lut = np.zeros(256, dtype=np.uint8)

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        for x in range(x1, min(x2 + 1, 256)):
            if x2 != x1:
                y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            else:
                y = y1
            lut[x] = int(np.clip(y, 0, 255))

    return cv2.LUT(image, lut), lut


image = np.random.randint(40, 220, (300, 400), dtype=np.uint8)

transformations = [
    [(0, 0), (255, 255)],
    [(0, 0), (60, 20), (180, 235), (255, 255)],
    [(0, 0), (100, 50), (150, 200), (255, 255)],
    [(0, 0), (50, 0), (200, 255), (255, 255)],
]
titles = ['恒等', '中间增强', 'S曲线', '高对比度']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('分段线性对比度增强', fontsize=14, fontweight='bold')

for i, (points, title) in enumerate(zip(transformations, titles)):
    result, lut = piecewise_linear_transform(image, points)
    axes[0, i].imshow(result, cmap='gray')
    axes[0, i].set_title(title, fontsize=11)
    axes[0, i].axis('off')

    axes[1, i].plot(range(256), lut, 'b-', linewidth=2)
    axes[1, i].plot([p[0] for p in points], [p[1] for p in points], 'ro', markersize=8)
    axes[1, i].plot([0, 255], [0, 255], 'k--', alpha=0.3)
    axes[1, i].set_xlim([0, 255])
    axes[1, i].set_ylim([0, 255])
    axes[1, i].set_title(f'{title}映射曲线')
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_piecewise_linear.png'), dpi=150, bbox_inches='tight')
plt.show()
