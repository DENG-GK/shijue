"""
示例4：对比不同形状结构元素产生的梯度效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_shape():
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (120, 120), 255, -1)
    return img

original = create_shape()

kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

gradient_rect = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, kernel_rect)
gradient_cross = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, kernel_cross)
gradient_ellipse = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, kernel_ellipse)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('原图', fontsize=11)
axes[0, 0].axis('off')
axes[0, 1].imshow(kernel_rect, cmap='gray')
axes[0, 1].set_title('矩形核', fontsize=11)
axes[0, 1].axis('off')
axes[0, 2].imshow(kernel_cross, cmap='gray')
axes[0, 2].set_title('十字核', fontsize=11)
axes[0, 2].axis('off')
axes[0, 3].imshow(kernel_ellipse, cmap='gray')
axes[0, 3].set_title('椭圆核', fontsize=11)
axes[0, 3].axis('off')
axes[1, 0].imshow(original, cmap='gray')
axes[1, 0].set_title('原图', fontsize=11)
axes[1, 0].axis('off')
axes[1, 1].imshow(gradient_rect, cmap='gray')
axes[1, 1].set_title('矩形核梯度', fontsize=11)
axes[1, 1].axis('off')
axes[1, 2].imshow(gradient_cross, cmap='gray')
axes[1, 2].set_title('十字核梯度', fontsize=11)
axes[1, 2].axis('off')
axes[1, 3].imshow(gradient_ellipse, cmap='gray')
axes[1, 3].set_title('椭圆核梯度', fontsize=11)
axes[1, 3].axis('off')
plt.suptitle('不同形状结构元素的梯度效果', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gradient_kernel_shape.png', dpi=150)
plt.show()
