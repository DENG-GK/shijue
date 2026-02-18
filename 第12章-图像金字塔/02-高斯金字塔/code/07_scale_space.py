"""
示例7：尺度空间分析
- 高斯金字塔构建尺度空间
- 每层Laplacian响应
- DoG (Difference of Gaussians)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建包含多尺度特征的图像
image = np.zeros((256, 256), dtype=np.uint8)
cv2.rectangle(image, (20, 20), (236, 236), 100, 3)
cv2.circle(image, (128, 128), 40, 200, -1)
cv2.circle(image, (80, 80), 10, 230, -1)
cv2.circle(image, (180, 80), 10, 230, -1)
for i in range(10):
    for j in range(10):
        cv2.circle(image, (30 + i * 20, 160 + j * 8), 2, 255, -1)

n_levels = 5
pyramid = [image.astype(np.float64)]
current = image.astype(np.float64)
for i in range(n_levels - 1):
    current = cv2.pyrDown(current)
    pyramid.append(current)

# Laplacian
laplacians = [cv2.Laplacian(level, cv2.CV_64F) for level in pyramid]

# DoG
dogs = []
for i in range(len(pyramid) - 1):
    upsampled = cv2.pyrUp(pyramid[i + 1])
    if upsampled.shape != pyramid[i].shape:
        upsampled = cv2.resize(upsampled, (pyramid[i].shape[1], pyramid[i].shape[0]))
    dogs.append(pyramid[i] - upsampled)

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle('尺度空间分析', fontsize=14, fontweight='bold')

for i, level in enumerate(pyramid):
    axes[0, i].imshow(level, cmap='gray')
    axes[0, i].set_title(f'Scale {i}')
    axes[0, i].axis('off')

for i, lap in enumerate(laplacians):
    axes[1, i].imshow(lap, cmap='RdBu_r')
    axes[1, i].set_title(f'Laplacian {i}')
    axes[1, i].axis('off')

for i, dog in enumerate(dogs):
    axes[2, i].imshow(dog, cmap='RdBu_r')
    axes[2, i].set_title(f'DoG {i}-{i + 1}')
    axes[2, i].axis('off')
axes[2, len(dogs)].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_scale_space.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n尺度空间特征分析:")
for i, lap in enumerate(laplacians):
    energy = np.sum(np.abs(lap))
    max_resp = np.max(np.abs(lap))
    print(f"Level {i}: 能量={energy:.0f}, 最大响应={max_resp:.2f}")
