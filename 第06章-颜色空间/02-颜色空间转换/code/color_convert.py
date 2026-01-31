# -*- coding: utf-8 -*-
"""颜色空间转换示例"""
import cv2
import numpy as np
import os

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
# 读取测试图片
SRC_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "01-颜色空间理论", "images", "color_shapes.png")

if not os.path.exists(SRC_IMG):
    print("请先运行 01-颜色空间理论/code/generate_images.py 生成测试图片")
    # 创建备用图片
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 80, (0, 0, 255), -1)
    cv2.circle(img, (400, 200), 80, (0, 255, 0), -1)
else:
    img = cv2.imread(SRC_IMG)

os.makedirs(IMAGES_DIR, exist_ok=True)

# BGR → 灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(IMAGES_DIR, "gray.png"), gray)
print(f"灰度图 shape: {gray.shape}")

# BGR → RGB（用于 matplotlib 显示）
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"RGB shape: {rgb.shape}")

# BGR → HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite(os.path.join(IMAGES_DIR, "hsv.png"), hsv)
print(f"HSV shape: {hsv.shape}, H范围: 0-179, S范围: 0-255, V范围: 0-255")

# BGR → Lab
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
cv2.imwrite(os.path.join(IMAGES_DIR, "lab.png"), lab)
print(f"Lab shape: {lab.shape}")

# BGR → YCrCb
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
cv2.imwrite(os.path.join(IMAGES_DIR, "ycrcb.png"), ycrcb)
print(f"YCrCb shape: {ycrcb.shape}")

# 拼接对比图
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
comparison = np.hstack([img, gray_bgr, hsv, lab])
cv2.imwrite(os.path.join(IMAGES_DIR, "color_spaces_comparison.png"), comparison)
print(f"对比图已保存: {os.path.join(IMAGES_DIR, 'color_spaces_comparison.png')}")
