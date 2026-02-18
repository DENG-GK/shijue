"""
示例2：对比三种形态学梯度：基本梯度、内梯度、外梯度
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_shape():
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(img, (75, 75), 50, 255, -1)
    return img

original = create_shape()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

dilated = cv2.dilate(original, kernel)
eroded = cv2.erode(original, kernel)
basic_gradient = dilated - eroded
internal_gradient = original - eroded
external_gradient = dilated - original

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(dilated, cmap='gray')
axes[0, 1].set_title('膨胀', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(eroded, cmap='gray')
axes[0, 2].set_title('腐蚀', fontsize=11)
axes[0, 2].axis('off')

axes[1, 0].imshow(basic_gradient, cmap='gray')
axes[1, 0].set_title('基本梯度\n(膨胀 - 腐蚀)', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(internal_gradient, cmap='gray')
axes[1, 1].set_title('内梯度\n(原图 - 腐蚀)', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(external_gradient, cmap='gray')
axes[1, 2].set_title('外梯度\n(膨胀 - 原图)', fontsize=11)
axes[1, 2].axis('off')

plt.suptitle('三种形态学梯度对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('three_gradients.png', dpi=150)
plt.show()

print("边缘宽度分析：")
print(f"• 基本梯度边缘宽度: {np.sum(basic_gradient > 0)} 像素")
print(f"• 内梯度边缘宽度: {np.sum(internal_gradient > 0)} 像素")
print(f"• 外梯度边缘宽度: {np.sum(external_gradient > 0)} 像素")
