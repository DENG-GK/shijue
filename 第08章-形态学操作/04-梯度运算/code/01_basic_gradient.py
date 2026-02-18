"""
示例1：基本形态学梯度
使用 cv2.MORPH_GRADIENT 提取边缘
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_test_image():
    """创建包含多个形状的测试图像"""
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(img, (20, 30), (100, 110), 255, -1)
    cv2.circle(img, (180, 70), 45, 255, -1)
    pts = np.array([[250, 120], [200, 180], [280, 180]], dtype=np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img

original = create_test_image()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

gradient = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, kernel)
dilated = cv2.dilate(original, kernel)
eroded = cv2.erode(original, kernel)
gradient_manual = dilated - eroded

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(gradient, cmap='gray')
axes[0, 1].set_title('形态学梯度（边缘）', fontsize=12)
axes[0, 1].axis('off')

axes[1, 0].imshow(dilated, cmap='gray')
axes[1, 0].set_title('膨胀结果', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(eroded, cmap='gray')
axes[1, 1].set_title('腐蚀结果', fontsize=12)
axes[1, 1].axis('off')

plt.suptitle('形态学梯度 = 膨胀 - 腐蚀', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('morphological_gradient.png', dpi=150)
plt.show()

print("两种方法结果是否一致:", np.array_equal(gradient, gradient_manual))
