"""
示例3：演示结构元素大小如何影响边缘宽度
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
kernel_sizes = [3, 5, 9, 15]

fig, axes = plt.subplots(1, len(kernel_sizes) + 1, figsize=(15, 3))

axes[0].imshow(original, cmap='gray')
axes[0].set_title('原图', fontsize=11)
axes[0].axis('off')

for i, size in enumerate(kernel_sizes):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    gradient = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, kernel)
    axes[i + 1].imshow(gradient, cmap='gray')
    axes[i + 1].set_title(f'{size}×{size} 核\n边缘宽度≈{size-1}px', fontsize=10)
    axes[i + 1].axis('off')

plt.suptitle('结构元素大小对边缘宽度的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gradient_kernel_size.png', dpi=150)
plt.show()

print("结论：")
print("• 边缘宽度 ≈ 结构元素大小 - 1")
print("• 核越大，边缘越粗")
