"""
示例4：演示结构元素大小对开闭运算效果的影响
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_test():
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(img, (75, 75), 50, 255, -1)
    cv2.circle(img, (30, 30), 5, 255, -1)
    cv2.circle(img, (120, 30), 10, 255, -1)
    cv2.circle(img, (30, 120), 15, 255, -1)
    return img

original = create_test()
kernel_sizes = [5, 11, 21, 31]

fig, axes = plt.subplots(1, len(kernel_sizes) + 1, figsize=(15, 3))
axes[0].imshow(original, cmap='gray')
axes[0].set_title('原图', fontsize=11)
axes[0].axis('off')

for i, size in enumerate(kernel_sizes):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opened = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
    axes[i + 1].imshow(opened, cmap='gray')
    axes[i + 1].set_title(f'开运算 {size}×{size}', fontsize=11)
    axes[i + 1].axis('off')

plt.suptitle('结构元素大小对开运算的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('kernel_size_opening.png', dpi=150)
plt.show()

print("观察结论：")
print("• 小核（5×5）：只能去除最小的噪点")
print("• 大核（21×21+）：可以去除更大的噪点，但可能影响主物体")
