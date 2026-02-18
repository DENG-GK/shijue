"""
示例4：多尺度特征可视化
- 各层Canny边缘检测
- 各层Harris角点检测
- 特征数量统计
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建包含多尺度特征的图像
image = np.zeros((512, 512, 3), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (250, 250), (0, 100, 200), -1)
cv2.rectangle(image, (260, 260), (460, 460), (200, 100, 0), -1)
for i in range(4):
    cv2.circle(image, (100 + i * 100, 350), 30, (0, 255, 0), -1)
for i in range(16):
    cv2.circle(image, (50 + i * 30, 480), 8, (255, 255, 0), -1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
pyramid = [gray]
current = gray
for i in range(4):
    current = cv2.pyrDown(current)
    pyramid.append(current)

edges = [cv2.Canny(level, 50, 150) for level in pyramid]
corners = []
for level in pyramid:
    corner = cv2.cornerHarris(level.astype(np.float32), 2, 3, 0.04)
    corner = cv2.dilate(corner, None)
    corner_img = np.zeros_like(level)
    corner_img[corner > 0.01 * corner.max()] = 255
    corners.append(corner_img)

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle('多尺度特征可视化', fontsize=14, fontweight='bold')

for i, level in enumerate(pyramid):
    axes[0, i].imshow(level, cmap='gray')
    axes[0, i].set_title(f'Level {i}\n{level.shape[1]}×{level.shape[0]}')
    axes[0, i].axis('off')

for i, edge in enumerate(edges):
    axes[1, i].imshow(edge, cmap='gray')
    axes[1, i].set_title(f'边缘 L{i}')
    axes[1, i].axis('off')

for i, corner in enumerate(corners):
    axes[2, i].imshow(corner, cmap='gray')
    axes[2, i].set_title(f'角点 L{i}')
    axes[2, i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_multiscale_features.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n各层特征数量:")
for i, (edge, corner) in enumerate(zip(edges, corners)):
    print(f"Level {i}: 边缘像素={np.sum(edge > 0):,}, 角点像素={np.sum(corner > 0):,}")
