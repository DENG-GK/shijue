# -*- coding: utf-8 -*-
"""图像缩放示例"""
import cv2, numpy as np, os
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# 生成测试图像
img = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.circle(img, (300, 200), 100, (0, 255, 0), -1)
cv2.rectangle(img, (100, 50), (500, 350), (255, 255, 0), 2)
cv2.putText(img, "Resize Test", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)

# 指定尺寸缩放
small = cv2.resize(img, (300, 200))
large = cv2.resize(img, (900, 600))
print(f"原图: {img.shape}, 缩小: {small.shape}, 放大: {large.shape}")

# 按比例缩放
half = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
double = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
print(f"0.5x: {half.shape}, 2.0x: {double.shape}")

cv2.imwrite(os.path.join(IMAGES_DIR, "original.png"), img)
cv2.imwrite(os.path.join(IMAGES_DIR, "small.png"), small)
cv2.imwrite(os.path.join(IMAGES_DIR, "half.png"), half)
print("缩放结果已保存")
