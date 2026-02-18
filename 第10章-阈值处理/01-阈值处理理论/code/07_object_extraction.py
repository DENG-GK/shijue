"""
示例7：使用阈值处理提取目标
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_bright_objects(image, threshold_value=200):
    """提取图像中的亮色目标"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(image.shape) == 3:
        result = image.copy()
    else:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return binary, result, len(contours)

test_img = np.zeros((400, 600), dtype=np.uint8)
test_img[:] = 30

cv2.circle(test_img, (100, 200), 50, 220, -1)
cv2.rectangle(test_img, (200, 100), (350, 250), 240, -1)
cv2.ellipse(test_img, (500, 200), (60, 40), 30, 0, 360, 210, -1)

noise = np.random.normal(0, 10, test_img.shape)
test_img = np.clip(test_img + noise, 0, 255).astype(np.uint8)

binary, result, count = extract_bright_objects(test_img, 180)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(test_img, cmap='gray')
axes[0].set_title('Original Image', fontsize=12)
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('Binary Mask', fontsize=12)
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'Detected Objects: {count}', fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(f"检测到 {count} 个目标")
