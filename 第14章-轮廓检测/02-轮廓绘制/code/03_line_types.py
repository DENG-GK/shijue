"""
示例3：线条类型
- LINE_4 / LINE_8 / LINE_AA
- 不同线宽效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建圆形轮廓
img = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(img, (100, 100), 70, 255, -1)
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

line_types = {
    'LINE_4': cv2.LINE_4,
    'LINE_8': cv2.LINE_8,
    'LINE_AA': cv2.LINE_AA,
}

thicknesses = [1, 2, 4]

fig, axes = plt.subplots(len(thicknesses), len(line_types), figsize=(12, 12))
fig.suptitle('线条类型与线宽', fontsize=14, fontweight='bold')

for row, thickness in enumerate(thicknesses):
    for col, (name, line_type) in enumerate(line_types.items()):
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.drawContours(canvas, contours, 0, (0, 255, 0), thickness, line_type)
        axes[row, col].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f'{name}, t={thickness}')
        axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_line_types.png'), dpi=150, bbox_inches='tight')
plt.show()
