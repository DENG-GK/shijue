"""
示例1：基本高斯金字塔构建
- build_gaussian_pyramid函数
- cv2.pyrDown逐层构建
- 可视化各层图像和堆叠视图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def build_gaussian_pyramid(image, levels=6):
    """构建高斯金字塔"""
    pyramid = [image]
    current = image
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
        print(f"Level {i + 1}: {current.shape}")
    return pyramid


# 创建测试图像
image = np.zeros((512, 512, 3), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (200, 200), (0, 0, 255), -1)
cv2.circle(image, (350, 150), 80, (0, 255, 0), -1)
cv2.ellipse(image, (256, 350), (100, 50), 45, 0, 360, (255, 0, 0), -1)
cv2.putText(image, 'Gaussian Pyramid', (100, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

print(f"原始图像: {image.shape}")
pyramid = build_gaussian_pyramid(image, levels=6)

fig = plt.figure(figsize=(18, 8))
fig.suptitle('高斯金字塔', fontsize=14, fontweight='bold')

# 各层展示
for i, level in enumerate(pyramid):
    ax = fig.add_subplot(2, 6, i + 1)
    ax.imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Level {i}\n{level.shape[1]}×{level.shape[0]}', fontsize=9)
    ax.axis('off')

# 堆叠视图
max_height = pyramid[0].shape[0]
stacked_width = sum(p.shape[1] for p in pyramid) + 10 * (len(pyramid) - 1)
stacked = np.ones((max_height, stacked_width, 3), dtype=np.uint8) * 255

x_offset = 0
for level in pyramid:
    h, w = level.shape[:2]
    y_offset = max_height - h
    stacked[y_offset:y_offset + h, x_offset:x_offset + w] = level
    x_offset += w + 10

ax = fig.add_subplot(2, 1, 2)
ax.imshow(cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB))
ax.set_title('高斯金字塔堆叠视图')
ax.axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_gaussian_pyramid_basic.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n总层数: {len(pyramid)}")
