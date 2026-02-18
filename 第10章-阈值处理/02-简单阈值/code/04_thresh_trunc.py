"""
THRESH_TRUNC 截断阈值
大于阈值的像素被截断为阈值，小于等于阈值的保持不变
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建渐变图像 =====================

gradient = np.tile(np.arange(256, dtype=np.uint8), (100, 1))

# ===================== 不同阈值的截断效果 =====================

thresholds = [64, 128, 192]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# 原图
axes[0].imshow(gradient, cmap='gray')
axes[0].set_title('原始渐变 (0 → 255)', fontsize=11)
axes[0].set_xlabel('像素值')
axes[0].set_yticks([])

# 不同阈值的截断效果
for i, T in enumerate(thresholds, 1):
    ret, result = cv2.threshold(gradient, T, 255, cv2.THRESH_TRUNC)
    axes[i].imshow(result, cmap='gray')
    axes[i].set_title(f'THRESH_TRUNC (T={T})\n最大值变为 {T}', fontsize=11)
    axes[i].set_xlabel('像素值')
    axes[i].set_yticks([])
    axes[i].axvline(x=T, color='r', linestyle='--', linewidth=2)

plt.suptitle('THRESH_TRUNC 截断效果', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('thresh_trunc_images.png', dpi=150, bbox_inches='tight')
plt.show()

# ===================== 像素值变化曲线 =====================

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(256)
ax.plot(x, x, 'b-', label='原始', linewidth=2)

for T in thresholds:
    y = np.minimum(x, T)
    ax.plot(x, y, '--', label=f'TRUNC T={T}', linewidth=2)

ax.set_xlabel('输入像素值', fontsize=12)
ax.set_ylabel('输出像素值', fontsize=12)
ax.set_title('THRESH_TRUNC: 像素值变换曲线', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)

plt.tight_layout()
plt.savefig('thresh_trunc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("THRESH_TRUNC 应用场景：")
print("- 限制图像的最大亮度")
print("- 处理过曝图像")
print("- 高光区域压缩")
