"""
THRESH_BINARY vs THRESH_BINARY_INV 对比
适合处理白底黑字或黑底白字的场景
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建包含文字的模拟图像 =====================

def create_text_image():
    """创建白底黑字的模拟图像"""
    img = np.ones((200, 400), dtype=np.uint8) * 240  # 浅色背景

    # 模拟文字（深色）
    cv2.putText(img, "Hello OpenCV!", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 2, 30, 3)

    return img

img = create_text_image()
T = 128

# ===================== 两种二值化方式 =====================

ret1, binary = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
ret2, binary_inv = cv2.threshold(img, T, 255, cv2.THRESH_BINARY_INV)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像\n(浅色背景，深色文字)', fontsize=11)
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('THRESH_BINARY\n(深色文字变黑)', fontsize=11)
axes[1].axis('off')

axes[2].imshow(binary_inv, cmap='gray')
axes[2].set_title('THRESH_BINARY_INV\n(深色文字变白)', fontsize=11)
axes[2].axis('off')

plt.suptitle('BINARY vs BINARY_INV 对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('binary_vs_binary_inv.png', dpi=150, bbox_inches='tight')
plt.show()

print("应用场景对比：")
print("- THRESH_BINARY: 适合提取浅色目标（目标比背景亮）")
print("- THRESH_BINARY_INV: 适合提取深色目标（目标比背景暗，如文字）")
