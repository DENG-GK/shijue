"""
示例4：层次绘制
- 利用hierarchy参数控制绘制层级
- maxLevel参数效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建嵌套形状
img = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img, (20, 20), (280, 280), 255, -1)
cv2.rectangle(img, (40, 40), (260, 260), 0, -1)
cv2.circle(img, (150, 150), 70, 255, -1)
cv2.circle(img, (150, 150), 40, 0, -1)
cv2.circle(img, (150, 150), 15, 255, -1)

contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('层次绘制 (maxLevel参数)', fontsize=14, fontweight='bold')

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

# 不同maxLevel
for idx, max_level in enumerate([0, 1, 2, 3]):
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.drawContours(canvas, contours, 0, (0, 255, 0), 2, cv2.LINE_8, hierarchy, max_level)
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    axes[row, col].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f'contourIdx=0, maxLevel={max_level}')
    axes[row, col].axis('off')

# 全部轮廓不同颜色
canvas = np.zeros((300, 300, 3), dtype=np.uint8)
level_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
for i in range(len(contours)):
    level = 0
    parent = hierarchy[0][i][3]
    while parent != -1:
        level += 1
        parent = hierarchy[0][parent][3]
    cv2.drawContours(canvas, contours, i, level_colors[level % 5], 2)
axes[1, 2].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('按层级着色')
axes[1, 2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_hierarchy_draw.png'), dpi=150, bbox_inches='tight')
plt.show()
