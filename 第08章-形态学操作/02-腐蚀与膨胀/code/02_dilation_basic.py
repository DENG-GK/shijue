"""
示例2：膨胀操作基础
演示 cv2.dilate() 的使用方法和效果
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

    # 带有内部空洞的矩形
    cv2.rectangle(img, (20, 20), (100, 100), 255, -1)
    cv2.rectangle(img, (45, 45), (75, 75), 0, -1)  # 空洞

    # 两个分离的小圆（可能会连接）
    cv2.circle(img, (150, 60), 20, 255, -1)
    cv2.circle(img, (195, 60), 20, 255, -1)

    # 细线条
    cv2.line(img, (20, 150), (280, 150), 255, 2)

    # 分散的小点
    for pos in [(240, 40), (260, 60), (250, 80)]:
        cv2.circle(img, pos, 3, 255, -1)

    return img

# ===================== 膨胀操作 =====================

original = create_test_image()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(original, kernel, iterations=1)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(original, cmap='gray')
axes[0].set_title('原始图像', fontsize=14)
axes[0].axis('off')

axes[1].imshow(dilated, cmap='gray')
axes[1].set_title('膨胀后 (5×5 矩形核)', fontsize=14)
axes[1].axis('off')

plt.suptitle('膨胀操作效果', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('dilation_basic.png', dpi=150)
plt.show()

print("观察结果：")
print("• 所有物体都变大了（边缘向外扩张）")
print("• 靠近的物体可能连接在一起")
print("• 内部小空洞可能被填补")
print("• 细线条变粗了")
