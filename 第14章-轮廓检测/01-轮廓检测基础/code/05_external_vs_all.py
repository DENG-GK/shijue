"""
示例5：外部轮廓 vs 所有轮廓
- RETR_EXTERNAL 仅外部
- RETR_TREE 全部层次
- 实际应用差异
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建嵌套形状
img = np.zeros((300, 400), dtype=np.uint8)

# 甜甜圈
cv2.circle(img, (100, 150), 80, 255, -1)
cv2.circle(img, (100, 150), 40, 0, -1)

# 嵌套矩形
cv2.rectangle(img, (220, 30), (380, 270), 255, -1)
cv2.rectangle(img, (240, 50), (360, 250), 0, -1)
cv2.rectangle(img, (260, 70), (340, 230), 255, -1)
cv2.rectangle(img, (280, 90), (320, 210), 0, -1)

# RETR_EXTERNAL
contours_ext, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
canvas_ext = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
cv2.drawContours(canvas_ext, contours_ext, -1, (0, 255, 0), 2)

# RETR_LIST
contours_list, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
canvas_list = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255)]
for i, cnt in enumerate(contours_list):
    cv2.drawContours(canvas_list, [cnt], 0, colors[i % len(colors)], 2)

# RETR_TREE
contours_tree, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
canvas_tree = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
level_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
for i in range(len(contours_tree)):
    level = 0
    parent = hierarchy[0][i][3]
    while parent != -1:
        level += 1
        parent = hierarchy[0][parent][3]
    cv2.drawContours(canvas_tree, contours_tree, i, level_colors[level % 4], 2)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('外部轮廓 vs 所有轮廓', fontsize=14, fontweight='bold')

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(canvas_ext, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'RETR_EXTERNAL\n{len(contours_ext)} 个轮廓')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(canvas_list, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'RETR_LIST\n{len(contours_list)} 个轮廓')
axes[2].axis('off')

axes[3].imshow(cv2.cvtColor(canvas_tree, cv2.COLOR_BGR2RGB))
axes[3].set_title(f'RETR_TREE\n{len(contours_tree)} 个轮廓')
axes[3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_external_vs_all.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"RETR_EXTERNAL: {len(contours_ext)} 个轮廓")
print(f"RETR_LIST:     {len(contours_list)} 个轮廓")
print(f"RETR_TREE:     {len(contours_tree)} 个轮廓")
