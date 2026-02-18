"""
示例1：基本轮廓绘制
- drawContours 基本用法
- 绘制全部/单个轮廓
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (30, 30), (150, 130), 255, -1)
cv2.circle(img, (260, 80), 50, 255, -1)
cv2.ellipse(img, (160, 230), (70, 40), 0, 0, 360, 255, -1)
cv2.rectangle(img, (300, 180), (380, 280), 255, -1)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('基本轮廓绘制', fontsize=14, fontweight='bold')

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

# 绘制全部轮廓
canvas1 = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.drawContours(canvas1, contours, -1, (0, 255, 0), 2)
axes[0, 1].imshow(cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('全部轮廓 (contourIdx=-1)')
axes[0, 1].axis('off')

# 逐个绘制不同颜色
canvas2 = np.zeros((300, 400, 3), dtype=np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
for i, cnt in enumerate(contours):
    cv2.drawContours(canvas2, contours, i, colors[i % len(colors)], 2)
axes[0, 2].imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('逐个绘制 (不同颜色)')
axes[0, 2].axis('off')

# 不同线宽
for idx, thickness in enumerate([1, 3, -1]):
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), thickness)
    label = f'thickness={thickness}' if thickness > 0 else 'thickness=-1 (填充)'
    axes[1, idx].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[1, idx].set_title(label)
    axes[1, idx].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_basic_draw.png'), dpi=150, bbox_inches='tight')
plt.show()
