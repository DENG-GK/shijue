"""
示例5：对比度切片（灰度级切片）
- 高亮特定灰度范围的像素
- 二值切片 vs 保留背景切片
- 多级伪彩色切片
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def intensity_level_slicing(image, low, high, highlight_value=255, method='highlight'):
    """灰度级切片"""
    if method == 'binary':
        result = np.zeros_like(image)
        mask = (image >= low) & (image <= high)
        result[mask] = highlight_value
    else:
        result = image.copy()
        mask = (image >= low) & (image <= high)
        result[mask] = highlight_value
    return result


# 创建具有不同灰度级的测试图像
image = np.zeros((300, 400), dtype=np.uint8)
for i in range(8):
    image[:, i * 50:(i + 1) * 50] = i * 32

ranges = [(50, 100), (100, 150), (150, 200)]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('灰度级切片', fontsize=14, fontweight='bold')

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('原图')
axes[0, 0].axis('off')

for i, (low, high) in enumerate(ranges):
    result = intensity_level_slicing(image, low, high, method='highlight')
    axes[0, i + 1].imshow(result, cmap='gray')
    axes[0, i + 1].set_title(f'高亮 [{low}-{high}]')
    axes[0, i + 1].axis('off')

for i, (low, high) in enumerate(ranges):
    result = intensity_level_slicing(image, low, high, method='binary')
    axes[1, i].imshow(result, cmap='gray')
    axes[1, i].set_title(f'二值 [{low}-{high}]')
    axes[1, i].axis('off')

# 伪彩色
pseudo_color = np.zeros((*image.shape, 3), dtype=np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
levels = [(0, 64), (64, 128), (128, 192), (192, 256)]
for (low, high), color in zip(levels, colors):
    mask = (image >= low) & (image < high)
    pseudo_color[mask] = color

axes[1, 3].imshow(pseudo_color)
axes[1, 3].set_title('多级伪彩色')
axes[1, 3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_intensity_slicing.png'), dpi=150, bbox_inches='tight')
plt.show()
