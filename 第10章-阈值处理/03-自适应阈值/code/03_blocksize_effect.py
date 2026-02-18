"""
blockSize 参数的影响
展示不同邻域大小对自适应阈值结果的影响
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

img = np.zeros((300, 400), dtype=np.uint8)
img[:] = 180

# 添加不同大小的文字和细节
cv2.putText(img, "Block Size Test", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 30, 3)
cv2.putText(img, "Large text", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 40, 2)
cv2.putText(img, "Medium text here", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 50, 2)
cv2.putText(img, "Small tiny text", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 60, 1)

# 添加细线
cv2.line(img, (20, 260), (380, 260), 50, 1)
cv2.line(img, (20, 280), (380, 280), 50, 2)

# 添加光照渐变
for i in range(300):
    factor = 0.7 + 0.3 * (i / 300)
    img[i, :] = (img[i, :] * factor).astype(np.uint8)

# ===================== 测试不同的blockSize =====================

block_sizes = [3, 11, 31, 51, 101]
C = 5

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 原图
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

# 不同blockSize的结果
for i, bs in enumerate(block_sizes, 1):
    result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bs, C)
    axes[i].imshow(result, cmap='gray')
    axes[i].set_title(f'blockSize = {bs}', fontsize=12)
    axes[i].axis('off')

plt.suptitle('blockSize 参数的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('blocksize_effect.png', dpi=150, bbox_inches='tight')
plt.show()

print("blockSize 参数指南：")
print("━" * 50)
print("小值 (3-11):  对细节敏感，但可能产生更多噪声")
print("中值 (21-51): 平衡细节和噪声，通常是好选择")
print("大值 (>51):   平滑效果好，但可能丢失细节")
print("━" * 50)
print("\n提示：blockSize 应该大于要分割的目标的笔画宽度")
