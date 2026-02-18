"""
示例3：开运算和闭运算组合使用
同时去除噪点和填补空洞
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_complex_image():
    """创建同时有噪点和空洞的图像"""
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (140, 160), 255, -1)
    cv2.circle(img, (90, 100), 20, 0, -1)
    cv2.circle(img, (210, 100), 50, 255, -1)
    cv2.circle(img, (210, 100), 15, 0, -1)
    np.random.seed(123)
    for _ in range(50):
        x = np.random.randint(0, 300)
        y = np.random.randint(0, 200)
        cv2.circle(img, (x, y), np.random.randint(2, 5), 255, -1)
    return img

original = create_complex_image()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

opened = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
open_then_close = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

closed = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)
close_then_open = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].set_title('原图', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(opened, cmap='gray')
axes[0, 1].set_title('① 开运算（去噪）', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(open_then_close, cmap='gray')
axes[0, 2].set_title('② 闭运算（填洞）', fontsize=12)
axes[0, 2].axis('off')

axes[1, 0].imshow(original, cmap='gray')
axes[1, 0].set_title('原图', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(closed, cmap='gray')
axes[1, 1].set_title('① 闭运算（填洞）', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(close_then_open, cmap='gray')
axes[1, 2].set_title('② 开运算（去噪）', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle('开闭运算组合对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('open_close_combination.png', dpi=150)
plt.show()

print("对比结论：")
print("• 先开后闭：先去噪再填洞，边缘更平滑")
print("• 先闭后开：先填洞再去噪，可能保留一些噪点")
