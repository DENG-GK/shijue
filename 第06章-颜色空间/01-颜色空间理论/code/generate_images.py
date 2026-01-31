# -*- coding: utf-8 -*-
"""生成第06章测试图片 - 彩色圆环图"""
import cv2
import numpy as np
import os

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# 创建彩色测试图（多种颜色的圆和矩形）
img = np.zeros((400, 600, 3), dtype=np.uint8)

# 红色圆
cv2.circle(img, (100, 100), 60, (0, 0, 255), -1)
# 绿色圆
cv2.circle(img, (300, 100), 60, (0, 255, 0), -1)
# 蓝色圆
cv2.circle(img, (500, 100), 60, (255, 0, 0), -1)
# 黄色矩形
cv2.rectangle(img, (40, 220), (180, 360), (0, 255, 255), -1)
# 青色矩形
cv2.rectangle(img, (230, 220), (370, 360), (255, 255, 0), -1)
# 紫色矩形
cv2.rectangle(img, (420, 220), (560, 360), (255, 0, 255), -1)

cv2.imwrite(os.path.join(IMAGES_DIR, "color_shapes.png"), img)
print(f"测试图片已保存: {os.path.join(IMAGES_DIR, 'color_shapes.png')}")
