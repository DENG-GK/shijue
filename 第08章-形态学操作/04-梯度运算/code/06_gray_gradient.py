"""
示例6：形态学梯度在灰度图像上的应用
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_gray_image():
    img = np.zeros((200, 200), dtype=np.uint8)
    for r in range(80, 0, -1):
        gray_value = int(255 * (80 - r) / 80)
        cv2.circle(img, (100, 100), r, gray_value, -1)
    return img

gray = create_gray_image()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)
sobel = np.uint8(np.clip(sobel, 0, 255))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title('原始灰度图像', fontsize=12)
axes[0].axis('off')
axes[1].imshow(gradient, cmap='gray')
axes[1].set_title('形态学梯度', fontsize=12)
axes[1].axis('off')
axes[2].imshow(sobel, cmap='gray')
axes[2].set_title('Sobel 梯度（对比）', fontsize=12)
axes[2].axis('off')
plt.suptitle('灰度图像的形态学梯度', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gray_gradient.png', dpi=150)
plt.show()
