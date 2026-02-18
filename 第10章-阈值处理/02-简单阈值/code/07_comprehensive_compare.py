"""
五种阈值类型综合对比
使用同一张图像展示所有阈值类型的效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建综合测试图像 =====================

def create_comprehensive_test_image():
    img = np.zeros((250, 350), dtype=np.uint8)

    # 背景渐变
    for i in range(250):
        img[i, :] = int(30 + i * 0.4)

    # 添加不同亮度的形状
    cv2.circle(img, (80, 125), 50, 220, -1)    # 亮圆
    cv2.rectangle(img, (150, 50), (250, 200), 150, -1)  # 中灰矩形
    cv2.ellipse(img, (300, 125), (30, 60), 0, 0, 360, 80, -1)  # 暗椭圆

    return img

img = create_comprehensive_test_image()
T = 120

# ===================== 五种阈值类型 =====================

types_info = [
    (cv2.THRESH_BINARY, 'BINARY', '>T→255, ≤T→0'),
    (cv2.THRESH_BINARY_INV, 'BINARY_INV', '>T→0, ≤T→255'),
    (cv2.THRESH_TRUNC, 'TRUNC', '>T→T, ≤T→保持'),
    (cv2.THRESH_TOZERO, 'TOZERO', '>T→保持, ≤T→0'),
    (cv2.THRESH_TOZERO_INV, 'TOZERO_INV', '>T→0, ≤T→保持'),
]

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 原图
axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'原始图像\n阈值 T = {T}', fontsize=12)
axes[0].axis('off')

# 各种阈值类型
for i, (thresh_type, name, formula) in enumerate(types_info, 1):
    ret, result = cv2.threshold(img, T, 255, thresh_type)
    axes[i].imshow(result, cmap='gray')
    axes[i].set_title(f'{name}\n{formula}', fontsize=11)
    axes[i].axis('off')

plt.suptitle('五种阈值类型综合对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('threshold_comprehensive_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("五种阈值类型总结：")
print(f"  BINARY:      >T→255, ≤T→0      (完全二值化)")
print(f"  BINARY_INV:  >T→0, ≤T→255      (反二值化)")
print(f"  TRUNC:       >T→T, ≤T→保持     (截断高值)")
print(f"  TOZERO:      >T→保持, ≤T→0     (去除低值)")
print(f"  TOZERO_INV:  >T→0, ≤T→保持     (去除高值)")
