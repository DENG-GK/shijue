"""
示例2：闭运算基础
演示 cv2.morphologyEx() 配合 cv2.MORPH_CLOSE 的使用
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_holed_image():
    """创建一个带有空洞的二值图像"""
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (130, 170), 255, -1)
    cv2.circle(img, (200, 100), 60, 255, -1)
    cv2.circle(img, (80, 80), 15, 0, -1)
    cv2.circle(img, (80, 130), 10, 0, -1)
    cv2.rectangle(img, (50, 100), (65, 115), 0, -1)
    cv2.circle(img, (200, 100), 20, 0, -1)
    cv2.circle(img, (180, 80), 8, 0, -1)
    cv2.circle(img, (220, 120), 8, 0, -1)
    return img

holed = create_holed_image()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

closed = cv2.morphologyEx(holed, cv2.MORPH_CLOSE, kernel)
dilated = cv2.dilate(holed, kernel)
closed_manual = cv2.erode(dilated, kernel)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(holed, cmap='gray')
axes[0, 0].set_title('原图（带空洞）', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(closed, cmap='gray')
axes[0, 1].set_title('闭运算结果', fontsize=12)
axes[0, 1].axis('off')

axes[1, 0].imshow(dilated, cmap='gray')
axes[1, 0].set_title('中间步骤：膨胀', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(closed_manual, cmap='gray')
axes[1, 1].set_title('膨胀后腐蚀（= 闭运算）', fontsize=12)
axes[1, 1].axis('off')

plt.suptitle('闭运算填洞效果', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('closing_basic.png', dpi=150)
plt.show()

print("两种方法结果是否一致:", np.array_equal(closed, closed_manual))
