"""
示例6：实际应用 - 指纹图像增强
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_fingerprint_image():
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(10):
        y_base = 20 + i * 18
        pts = []
        for x in range(10, 190):
            y = y_base + int(10 * np.sin(x / 15 + i * 0.5))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], False, 255, 2)
    for _ in range(20):
        x = np.random.randint(20, 180)
        y = np.random.randint(20, 180)
        cv2.circle(img, (x, y), np.random.randint(3, 8), 0, -1)
    noise = np.random.random((200, 200))
    img[noise < 0.03] = 255
    img[noise > 0.97] = 0
    return img

fingerprint = create_fingerprint_image()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(fingerprint, cv2.MORPH_OPEN, kernel)
method1 = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
method2 = cv2.morphologyEx(fingerprint, cv2.MORPH_CLOSE, kernel_h)
method2 = cv2.morphologyEx(method2, cv2.MORPH_OPEN, kernel)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(fingerprint, cmap='gray')
axes[0].set_title('原始指纹图像', fontsize=11)
axes[0].axis('off')
axes[1].imshow(method1, cmap='gray')
axes[1].set_title('方案1：先开后闭', fontsize=11)
axes[1].axis('off')
axes[2].imshow(method2, cmap='gray')
axes[2].set_title('方案2：横向闭+开', fontsize=11)
axes[2].axis('off')
plt.suptitle('指纹图像增强', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fingerprint_enhancement.png', dpi=150)
plt.show()

print("技巧：使用横向核可更好连接指纹纹路")
