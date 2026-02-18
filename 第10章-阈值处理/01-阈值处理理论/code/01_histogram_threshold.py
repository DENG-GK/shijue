"""
示例1：理解图像直方图与阈值的关系
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建一个简单的测试图像
img = np.zeros((200, 200), dtype=np.uint8)
img[50:150, 50:150] = 200
img = cv2.GaussianBlur(img, (21, 21), 0)

noise = np.random.normal(0, 10, img.shape).astype(np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.plot(hist)
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.axvline(x=100, color='r', linestyle='--', label='Threshold=100')
plt.legend()

_, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

plt.subplot(133)
plt.imshow(binary, cmap='gray')
plt.title('After Thresholding (T=100)')
plt.axis('off')

plt.tight_layout()
plt.show()

print("直方图分析：")
print(f"- 图像像素范围: {img.min()} ~ {img.max()}")
print(f"- 直方图峰值位置: {np.argmax(hist)}")
print(f"- 选择的阈值: 100")
