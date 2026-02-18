"""
示例5：实际应用 - 使用开闭运算处理文档扫描图像
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_document_image():
    img = np.zeros((200, 400), dtype=np.uint8)
    cv2.rectangle(img, (30, 40), (180, 60), 255, -1)
    cv2.rectangle(img, (30, 80), (220, 100), 255, -1)
    cv2.rectangle(img, (30, 120), (190, 140), 255, -1)
    cv2.rectangle(img, (80, 40), (90, 60), 0, -1)
    cv2.rectangle(img, (140, 80), (150, 100), 0, -1)
    cv2.rectangle(img, (100, 120), (108, 140), 0, -1)
    np.random.seed(42)
    noise = np.random.random((200, 400))
    img[noise < 0.02] = 255
    img[(noise > 0.98) & (img == 255)] = 0
    for _ in range(30):
        x = np.random.randint(250, 380)
        y = np.random.randint(20, 180)
        cv2.circle(img, (x, y), np.random.randint(1, 4), 255, -1)
    return img

document = create_document_image()
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

step1 = cv2.morphologyEx(document, cv2.MORPH_OPEN, kernel_open)
step2 = cv2.morphologyEx(step1, cv2.MORPH_CLOSE, kernel_close)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(document, cmap='gray')
axes[0].set_title('原始扫描图像\n（有噪点和断裂）', fontsize=11)
axes[0].axis('off')
axes[1].imshow(step1, cmap='gray')
axes[1].set_title('开运算后\n（噪点去除）', fontsize=11)
axes[1].axis('off')
axes[2].imshow(step2, cmap='gray')
axes[2].set_title('闭运算后\n（断裂修复）', fontsize=11)
axes[2].axis('off')
plt.suptitle('文档图像处理实例', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('document_processing.png', dpi=150)
plt.show()

print("处理流程：1.开运算去噪 2.闭运算填补断裂")
