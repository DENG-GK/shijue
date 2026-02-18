"""
C 参数的影响
展示不同C值对自适应阈值结果的影响
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

img = np.ones((250, 350), dtype=np.uint8) * 180
cv2.putText(img, "Testing C", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 40, 3)
cv2.putText(img, "parameter", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 50, 2)
cv2.putText(img, "variation", (60, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 60, 2)

# 添加渐变
for j in range(350):
    factor = 0.6 + 0.4 * (j / 350)
    img[:, j] = (img[:, j] * factor).astype(np.uint8)

# 添加噪声
noise = np.random.normal(0, 8, img.shape)
img = np.clip(img + noise, 0, 255).astype(np.uint8)

# ===================== 测试不同的C值 =====================

C_values = [-5, 0, 5, 10, 20]
block_size = 21

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 原图
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

# 不同C值的结果
for i, c in enumerate(C_values, 1):
    result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, c)
    axes[i].imshow(result, cmap='gray')
    title = f'C = {c}'
    if c < 0:
        title += '\n(更多白色)'
    elif c > 10:
        title += '\n(更多黑色)'
    axes[i].set_title(title, fontsize=12)
    axes[i].axis('off')

plt.suptitle('C 参数的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('c_param_effect.png', dpi=150, bbox_inches='tight')
plt.show()

print("C 参数指南：")
print("━" * 50)
print("负值 (C<0):   阈值更低，更多像素变成白色")
print("零值 (C=0):   使用计算的局部均值作为阈值")
print("正值 (C>0):   阈值更高，更多像素变成黑色")
print("━" * 50)
print("\n典型值：2-10，根据图像对比度调整")
print("对比度低的图像：使用较小的C")
print("对比度高的图像：使用较大的C")
