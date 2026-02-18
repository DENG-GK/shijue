"""
示例1：腐蚀操作基础
演示 cv2.erode() 的使用方法和效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建一个有各种元素的测试图像"""
    img = np.zeros((200, 300), dtype=np.uint8)

    # 大矩形
    cv2.rectangle(img, (20, 20), (100, 100), 255, -1)

    # 圆形
    cv2.circle(img, (170, 60), 40, 255, -1)

    # 带有细连接的两个区域
    cv2.rectangle(img, (230, 30), (280, 90), 255, -1)
    cv2.rectangle(img, (250, 85), (260, 115), 255, -1)  # 细连接
    cv2.rectangle(img, (230, 110), (280, 170), 255, -1)

    # 一些小噪点
    for pos in [(50, 150), (80, 160), (110, 140), (140, 155)]:
        cv2.circle(img, pos, 4, 255, -1)

    return img

# ===================== 腐蚀操作 =====================

# 创建原始图像
original = create_test_image()

# 创建结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 执行腐蚀操作
eroded = cv2.erode(original, kernel, iterations=1)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(original, cmap='gray')
axes[0].set_title('原始图像', fontsize=14)
axes[0].axis('off')

axes[1].imshow(eroded, cmap='gray')
axes[1].set_title('腐蚀后 (5×5 矩形核)', fontsize=14)
axes[1].axis('off')

plt.suptitle('腐蚀操作效果', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('erosion_basic.png', dpi=150)
plt.show()

print("观察结果：")
print("• 所有物体都变小了（边缘向内收缩）")
print("• 细连接可能被断开")
print("• 小噪点可能消失")
