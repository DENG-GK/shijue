"""
示例5：使用形态学梯度提取物体轮廓
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_objects():
    img = np.zeros((250, 350), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (100, 100), 255, -1)
    cv2.circle(img, (170, 60), 45, 255, -1)
    cv2.ellipse(img, (280, 60), (40, 30), 0, 0, 360, 255, -1)
    pts1 = np.array([[50, 150], [20, 220], [80, 220]], dtype=np.int32)
    cv2.fillPoly(img, [pts1], 255)
    cv2.rectangle(img, (120, 140), (200, 230), 255, -1)
    pts2 = np.array([[280, 140], [320, 170], [305, 220],
                     [255, 220], [240, 170]], dtype=np.int32)
    cv2.fillPoly(img, [pts2], 255)
    return img

original = create_objects()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, kernel)

original_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
overlay = original_color.copy()
overlay[edges > 0] = [0, 0, 255]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(original, cmap='gray')
axes[0].set_title('原始二值图像', fontsize=12)
axes[0].axis('off')
axes[1].imshow(edges, cmap='gray')
axes[1].set_title('提取的边缘', fontsize=12)
axes[1].axis('off')
axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[2].set_title('边缘叠加显示（红色）', fontsize=12)
axes[2].axis('off')
plt.suptitle('形态学梯度轮廓提取', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('contour_extraction.png', dpi=150)
plt.show()
