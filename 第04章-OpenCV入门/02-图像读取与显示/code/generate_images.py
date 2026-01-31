# -*- coding: utf-8 -*-
"""
生成测试图像 - 用于图像读取与显示演示
"""
import numpy as np
import cv2
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# 生成彩色测试图
img = np.zeros((300, 400, 3), dtype=np.uint8)

# 彩色条纹
colors = [
    (255, 0, 0),    # 蓝
    (0, 255, 0),    # 绿
    (0, 0, 255),    # 红
    (255, 255, 0),  # 青
    (255, 0, 255),  # 品红
    (0, 255, 255),  # 黄
    (255, 255, 255),# 白
    (128, 128, 128),# 灰
]

stripe_w = img.shape[1] // len(colors)
for i, color in enumerate(colors):
    img[:, i*stripe_w:(i+1)*stripe_w] = color

# 添加文字
cv2.putText(img, "OpenCV Test Image", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imwrite(os.path.join(IMAGES_DIR, "test_color.png"), img)

# 生成灰度渐变图
gray = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (300, 1))
cv2.imwrite(os.path.join(IMAGES_DIR, "test_gray.png"), gray)

print("测试图像已生成到 images/ 目录")
