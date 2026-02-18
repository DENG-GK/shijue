"""
示例3：演示迭代次数对腐蚀和膨胀效果的影响
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
def create_circle():
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(img, (75, 75), 50, 255, -1)
    return img

original = create_circle()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# ===================== 不同迭代次数的腐蚀 =====================

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# 腐蚀
axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('原图', fontsize=11)
axes[0, 0].axis('off')

for i, iteration in enumerate([1, 2, 3, 5], start=1):
    eroded = cv2.erode(original, kernel, iterations=iteration)
    axes[0, i].imshow(eroded, cmap='gray')
    axes[0, i].set_title(f'腐蚀 {iteration} 次', fontsize=11)
    axes[0, i].axis('off')

# 膨胀
axes[1, 0].imshow(original, cmap='gray')
axes[1, 0].set_title('原图', fontsize=11)
axes[1, 0].axis('off')

for i, iteration in enumerate([1, 2, 3, 5], start=1):
    dilated = cv2.dilate(original, kernel, iterations=iteration)
    axes[1, i].imshow(dilated, cmap='gray')
    axes[1, i].set_title(f'膨胀 {iteration} 次', fontsize=11)
    axes[1, i].axis('off')

# 添加行标签
fig.text(0.02, 0.75, '腐蚀', fontsize=14, fontweight='bold', rotation=90, va='center')
fig.text(0.02, 0.25, '膨胀', fontsize=14, fontweight='bold', rotation=90, va='center')

plt.suptitle('迭代次数对形态学操作的影响', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(left=0.05)
plt.savefig('iterations_effect.png', dpi=150)
plt.show()

print("观察结论：")
print("• 迭代次数越多，效果越明显")
print("• 每次迭代大约收缩/扩张 (kernel_size-1)/2 个像素")
print("• 迭代次数过多可能导致物体消失或过度膨胀")
