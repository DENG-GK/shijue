"""
示例4：在真实图像上对比五种阈值类型
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_sample_image():
    """创建一个包含渐变和形状的示例图像"""
    img = np.zeros((300, 400), dtype=np.uint8)
    for i in range(300):
        img[i, :] = int(50 + i * 0.3)
    cv2.circle(img, (100, 150), 60, 255, -1)
    cv2.rectangle(img, (200, 80), (350, 220), 200, -1)
    cv2.ellipse(img, (280, 150), (50, 80), 45, 0, 360, 30, -1)
    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img

img = create_sample_image()

threshold_types = [
    (cv2.THRESH_BINARY, 'BINARY'),
    (cv2.THRESH_BINARY_INV, 'BINARY_INV'),
    (cv2.THRESH_TRUNC, 'TRUNC'),
    (cv2.THRESH_TOZERO, 'TOZERO'),
    (cv2.THRESH_TOZERO_INV, 'TOZERO_INV'),
]

T = 100

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'Original Image\nThreshold T = {T}', fontsize=12)
axes[0].axis('off')

for i, (thresh_type, name) in enumerate(threshold_types, 1):
    _, result = cv2.threshold(img, T, 255, thresh_type)
    axes[i].imshow(result, cmap='gray')
    axes[i].set_title(f'{name}', fontsize=12)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
