"""
示例1：线性拉伸（对比度拉伸）
- 最小-最大归一化映射到[0,255]
- 充分利用整个灰度范围
- 对比拉伸前后的直方图变化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def linear_contrast_stretch(image, out_min=0, out_max=255):
    """线性对比度拉伸"""
    in_min = np.min(image)
    in_max = np.max(image)
    if in_max == in_min:
        return np.full_like(image, (out_min + out_max) // 2)
    stretched = (image - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return np.clip(stretched, out_min, out_max).astype(np.uint8)


# 创建低对比度测试图像
image = np.random.randint(80, 180, (300, 400), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (150, 150), 100, -1)
cv2.circle(image, (300, 150), 60, 160, -1)

stretched = linear_contrast_stretch(image)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('线性对比度拉伸', fontsize=14, fontweight='bold')

axes[0].imshow(image, cmap='gray')
axes[0].set_title(f'原图 (范围: {image.min()}-{image.max()})')
axes[0].axis('off')

axes[1].imshow(stretched, cmap='gray')
axes[1].set_title(f'拉伸后 (范围: {stretched.min()}-{stretched.max()})')
axes[1].axis('off')

axes[2].hist(image.flatten(), bins=256, range=[0, 256], alpha=0.5, label='原始')
axes[2].hist(stretched.flatten(), bins=256, range=[0, 256], alpha=0.5, label='拉伸后')
axes[2].set_title('直方图对比')
axes[2].legend()

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_linear_stretch.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"原始范围: [{image.min()}, {image.max()}]")
print(f"拉伸范围: [{stretched.min()}, {stretched.max()}]")
