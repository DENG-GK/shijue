"""
THRESH_TOZERO_INV 反置零阈值
大于阈值的像素变为0，小于等于阈值的保持不变
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建渐变图像 =====================

gradient = np.tile(np.arange(256, dtype=np.uint8), (100, 1))

thresholds = [64, 128, 192]

# ===================== 图像效果展示 =====================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# 原图
axes[0].imshow(gradient, cmap='gray')
axes[0].set_title('原始渐变 (0 → 255)', fontsize=11)
axes[0].set_xlabel('像素值')
axes[0].set_yticks([])

# 不同阈值的反置零效果
for i, T in enumerate(thresholds, 1):
    ret, result = cv2.threshold(gradient, T, 255, cv2.THRESH_TOZERO_INV)
    axes[i].imshow(result, cmap='gray')
    axes[i].set_title(f'THRESH_TOZERO_INV (T={T})\n值 > {T} 变为 0', fontsize=11)
    axes[i].set_xlabel('像素值')
    axes[i].set_yticks([])
    axes[i].axvline(x=T, color='r', linestyle='--', linewidth=2)

plt.suptitle('THRESH_TOZERO_INV 反置零效果', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('thresh_tozero_inv_images.png', dpi=150, bbox_inches='tight')
plt.show()

# ===================== 像素值变化曲线 =====================

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(256)
ax.plot(x, x, 'b-', label='原始', linewidth=2)

for T in thresholds:
    y = np.where(x > T, 0, x)
    ax.plot(x, y, '--', label=f'TOZERO_INV T={T}', linewidth=2)

ax.set_xlabel('输入像素值', fontsize=12)
ax.set_ylabel('输出像素值', fontsize=12)
ax.set_title('THRESH_TOZERO_INV: 像素值变换曲线', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)

plt.tight_layout()
plt.savefig('thresh_tozero_inv_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("THRESH_TOZERO_INV 应用场景：")
print("- 去除图像中的高光区域")
print("- 提取暗部细节同时保留原始亮度信息")
print("- 高光抑制")
