"""
示例4：轮廓层次结构
- hierarchy数组解读
- [Next, Previous, First_Child, Parent]
- 树形层次可视化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建多层嵌套图像
img = np.zeros((400, 500), dtype=np.uint8)

# 外层矩形1
cv2.rectangle(img, (20, 20), (220, 380), 255, -1)
cv2.rectangle(img, (40, 40), (200, 360), 0, -1)
# 内部小形状
cv2.circle(img, (120, 120), 30, 255, -1)
cv2.rectangle(img, (60, 220), (180, 320), 255, -1)
cv2.rectangle(img, (80, 240), (160, 300), 0, -1)

# 外层矩形2
cv2.rectangle(img, (260, 20), (480, 200), 255, -1)
cv2.circle(img, (370, 110), 50, 0, -1)

# 独立圆
cv2.circle(img, (370, 320), 60, 255, -1)

contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(f"检测到 {len(contours)} 个轮廓")
print("\n层次结构: [Next, Prev, Child, Parent]")
print("-" * 60)
for i in range(len(contours)):
    h = hierarchy[0][i]
    level = 0
    parent = h[3]
    while parent != -1:
        level += 1
        parent = hierarchy[0][parent][3]
    indent = "  " * level
    print(f"{indent}轮廓{i}: {h.tolist()}, 面积={cv2.contourArea(contours[i]):.0f}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('轮廓层次结构', fontsize=14, fontweight='bold')

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

# 按层级着色
level_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for i in range(len(contours)):
    h = hierarchy[0][i]
    level = 0
    parent = h[3]
    while parent != -1:
        level += 1
        parent = hierarchy[0][parent][3]
    color = level_colors[level % len(level_colors)]
    cv2.drawContours(canvas, contours, i, color, 2)
    M = cv2.moments(contours[i])
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(canvas, f"{i}(L{level})", (cx - 15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[1].set_title('轮廓层级 (颜色=层级)')
axes[1].axis('off')

# 树形文本
tree_text = "层次结构树:\n"
for i in range(len(contours)):
    h = hierarchy[0][i]
    level = 0
    parent = h[3]
    while parent != -1:
        level += 1
        parent = hierarchy[0][parent][3]
    indent = "  " * level
    tree_text += f"{indent}[{i}] A={cv2.contourArea(contours[i]):.0f}\n"

axes[2].text(0.05, 0.95, tree_text, fontsize=9, family='monospace',
             verticalalignment='top', transform=axes[2].transAxes)
axes[2].set_title('层次树')
axes[2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_hierarchy.png'), dpi=150, bbox_inches='tight')
plt.show()
