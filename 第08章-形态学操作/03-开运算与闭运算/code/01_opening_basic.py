"""
示例1：开运算基础
演示 cv2.morphologyEx() 配合 cv2.MORPH_OPEN 的使用
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_noisy_image():
    """创建一个带有白色噪点的二值图像"""
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (130, 130), 255, -1)
    cv2.circle(img, (200, 80), 50, 255, -1)
    np.random.seed(42)
    for _ in range(80):
        x = np.random.randint(0, 300)
        y = np.random.randint(0, 200)
        r = np.random.randint(2, 5)
        cv2.circle(img, (x, y), r, 255, -1)
    return img

noisy = create_noisy_image()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)
eroded = cv2.erode(noisy, kernel)
opened_manual = cv2.dilate(eroded, kernel)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(noisy, cmap='gray')
axes[0, 0].set_title('原图（带噪点）', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(opened, cmap='gray')
axes[0, 1].set_title('开运算结果', fontsize=12)
axes[0, 1].axis('off')

axes[1, 0].imshow(eroded, cmap='gray')
axes[1, 0].set_title('中间步骤：腐蚀', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(opened_manual, cmap='gray')
axes[1, 1].set_title('腐蚀后膨胀（= 开运算）', fontsize=12)
axes[1, 1].axis('off')

plt.suptitle('开运算去噪效果', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('opening_basic.png', dpi=150)
plt.show()

print("两种方法结果是否一致:", np.array_equal(opened, opened_manual))
