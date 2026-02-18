"""
THRESH_BINARY 二值化
使用不同阈值进行二值化处理
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_image():
    img = np.zeros((300, 400), dtype=np.uint8)

    # 创建5个不同灰度级的区域
    regions = [(50, 50), (100, 130), (150, 210), (200, 290), (250, 370)]
    for gray_value, x in regions:
        cv2.rectangle(img, (x, 50), (x+60, 250), gray_value, -1)

    return img

img = create_test_image()

# ===================== 不同阈值的二值化 =====================

thresholds = [80, 120, 160, 200]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 原图
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像\n(灰度值: 50, 100, 150, 200, 250)', fontsize=10)
axes[0].axis('off')

# 不同阈值的结果
for i, T in enumerate(thresholds, 1):
    ret, binary = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    axes[i].imshow(binary, cmap='gray')
    axes[i].set_title(f'THRESH_BINARY\nT = {T}', fontsize=10)
    axes[i].axis('off')

axes[5].axis('off')

plt.suptitle('THRESH_BINARY 二值化效果', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('thresh_binary.png', dpi=150, bbox_inches='tight')
plt.show()

# 详细打印处理结果
T = 120
ret, binary = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
print(f"阈值 T = {T} 的处理结果：")
print(f"- 返回值 retval = {ret}")
print(f"- 灰度值 50  (< {T}): 变为 0")
print(f"- 灰度值 100 (< {T}): 变为 0")
print(f"- 灰度值 150 (> {T}): 变为 255")
print(f"- 灰度值 200 (> {T}): 变为 255")
print(f"- 灰度值 250 (> {T}): 变为 255")
