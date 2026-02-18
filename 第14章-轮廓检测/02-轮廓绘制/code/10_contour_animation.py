"""
示例10：轮廓逐步绘制
- 模拟逐步绘制过程
- 多帧展示
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建形状
img = np.zeros((300, 300), dtype=np.uint8)
pts = []
for i in range(36):
    angle = i * 10 * np.pi / 180
    r = 80 + 30 * np.sin(5 * angle)
    x = int(150 + r * np.cos(angle))
    y = int(150 + r * np.sin(angle))
    pts.append([x, y])
cv2.fillPoly(img, [np.array(pts)], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
total_pts = len(cnt)

# 逐步绘制帧
steps = [0.1, 0.25, 0.5, 0.75, 1.0]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('轮廓逐步绘制', fontsize=14, fontweight='bold')

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title(f'原始图像\n轮廓点数: {total_pts}')
axes[0, 0].axis('off')

for idx, ratio in enumerate(steps):
    n_pts = int(total_pts * ratio)
    partial = cnt[:n_pts]

    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    # 绘制已有部分
    for j in range(len(partial) - 1):
        pt1 = tuple(partial[j][0])
        pt2 = tuple(partial[j + 1][0])
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)
    # 标记当前点
    if len(partial) > 0:
        cv2.circle(canvas, tuple(partial[-1][0]), 4, (0, 0, 255), -1)
    # 标记起点
    cv2.circle(canvas, tuple(cnt[0][0]), 4, (255, 0, 0), -1)

    row = (idx + 1) // 3
    col = (idx + 1) % 3
    axes[row, col].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f'{ratio*100:.0f}% ({n_pts}/{total_pts}点)')
    axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_contour_animation.png'), dpi=150, bbox_inches='tight')
plt.show()
