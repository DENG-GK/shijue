# -*- coding: utf-8 -*-
"""图像裁剪示例"""
import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# 生成测试图像（渐变+色块）
img = np.zeros((400, 600, 3), dtype=np.uint8)
for i in range(400):
    img[i, :, 0] = int(i / 400 * 255)
for j in range(600):
    img[:, j, 2] = int(j / 600 * 255)
cv2.rectangle(img, (150, 100), (450, 300), (0, 255, 0), 2)

# 裁剪ROI
roi = img[100:300, 150:450]
print(f"原图: {img.shape}")
print(f"ROI: {roi.shape}")

cv2.imwrite(os.path.join(IMAGES_DIR, "original.png"), img)
cv2.imwrite(os.path.join(IMAGES_DIR, "roi.png"), roi)

# ROI复制到另一个位置
img_copy = img.copy()
img_copy[0:200, 0:300] = roi
cv2.imwrite(os.path.join(IMAGES_DIR, "roi_paste.png"), img_copy)
print("裁剪结果已保存")
