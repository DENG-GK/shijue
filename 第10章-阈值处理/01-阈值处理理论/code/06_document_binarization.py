"""
示例6：文档图像二值化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def document_binarization(image):
    """文档图像二值化处理"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return gray, binary_otsu, binary_adaptive

def create_document_image():
    """创建模拟的文档图像"""
    img = np.ones((400, 600), dtype=np.uint8) * 240

    for i, y in enumerate(range(50, 350, 40)):
        width = np.random.randint(200, 500)
        cv2.rectangle(img, (50, y), (50 + width, y + 20), 30, -1)

    rows, cols = img.shape
    gradient = np.zeros_like(img, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            gradient[i, j] = 1.0 - 0.3 * (i / rows) - 0.2 * (j / cols)

    img = (img * gradient).astype(np.uint8)

    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

doc_img = create_document_image()
gray, binary_otsu, binary_adaptive = document_binarization(doc_img)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Original Document\n(with uneven lighting)', fontsize=12)
axes[0].axis('off')

axes[1].imshow(binary_otsu, cmap='gray')
axes[1].set_title('Otsu Thresholding\n(global threshold)', fontsize=12)
axes[1].axis('off')

axes[2].imshow(binary_adaptive, cmap='gray')
axes[2].set_title('Adaptive Thresholding\n(local threshold)', fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("对比分析：")
print("- Otsu方法：使用全局阈值，在光照不均时效果较差")
print("- 自适应方法：使用局部阈值，能更好地处理光照不均的情况")
