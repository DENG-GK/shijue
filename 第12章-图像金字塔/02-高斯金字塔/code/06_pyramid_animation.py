"""
示例6：金字塔构建过程可视化
- 原图 → 高斯滤波 → 下采样步骤
- 与cv2.pyrDown对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.circle(image, (128, 128), 80, (0, 255, 0), -1)
cv2.rectangle(image, (30, 30), (100, 100), (255, 0, 0), -1)
cv2.rectangle(image, (156, 156), (226, 226), (0, 0, 255), -1)

# 构建金字塔
pyramid = [image]
current = image
for i in range(5):
    current = cv2.pyrDown(current)
    pyramid.append(current)

fig, axes = plt.subplots(2, 6, figsize=(18, 8))
fig.suptitle('金字塔构建过程', fontsize=14, fontweight='bold')

# 各层
for i, level in enumerate(pyramid):
    axes[0, i].imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'Level {i}: {level.shape[1]}×{level.shape[0]}', fontsize=9)
    axes[0, i].axis('off')

# 构建步骤演示
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
steps = [
    (gray, '步骤1: 原图'),
    (cv2.GaussianBlur(gray, (5, 5), 0), '步骤2: 高斯滤波'),
    (cv2.GaussianBlur(gray, (5, 5), 0)[::2, ::2], '步骤3: 下采样'),
]

blurred = cv2.GaussianBlur(gray, (5, 5), 0)[::2, ::2]
steps.append((cv2.GaussianBlur(blurred, (5, 5), 0), '步骤4: 再滤波'))
steps.append((cv2.GaussianBlur(blurred, (5, 5), 0)[::2, ::2], '步骤5: 再下采样'))

opencv_result = cv2.pyrDown(cv2.pyrDown(gray))
steps.append((opencv_result, 'OpenCV pyrDown×2'))

for i, (img, title) in enumerate(steps):
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].set_title(title, fontsize=9)
    axes[1, i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_pyramid_animation.png'), dpi=150, bbox_inches='tight')
plt.show()
