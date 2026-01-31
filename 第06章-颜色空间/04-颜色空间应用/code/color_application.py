# -*- coding: utf-8 -*-
"""颜色空间应用 - HSV颜色过滤示例"""
import cv2
import numpy as np
import os

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
SRC_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "01-颜色空间理论", "images", "color_shapes.png")

if not os.path.exists(SRC_IMG):
    print("请先运行 01-颜色空间理论/code/generate_images.py 生成测试图片")
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(img, (150, 200), 80, (0, 0, 255), -1)
    cv2.circle(img, (300, 200), 80, (0, 255, 0), -1)
    cv2.circle(img, (450, 200), 80, (255, 0, 0), -1)
else:
    img = cv2.imread(SRC_IMG)

os.makedirs(IMAGES_DIR, exist_ok=True)

# 转换到 HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# === 提取红色区域 ===
# 红色在 HSV 中跨越 H=0 边界，需要两个范围取并集
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = mask_red1 | mask_red2
result_red = cv2.bitwise_and(img, img, mask=mask_red)

cv2.imwrite(os.path.join(IMAGES_DIR, "mask_red.png"), mask_red)
cv2.imwrite(os.path.join(IMAGES_DIR, "result_red.png"), result_red)
print("红色提取完成")

# === 提取绿色区域 ===
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
result_green = cv2.bitwise_and(img, img, mask=mask_green)

cv2.imwrite(os.path.join(IMAGES_DIR, "mask_green.png"), mask_green)
cv2.imwrite(os.path.join(IMAGES_DIR, "result_green.png"), result_green)
print("绿色提取完成")

# === 提取蓝色区域 ===
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
result_blue = cv2.bitwise_and(img, img, mask=mask_blue)

cv2.imwrite(os.path.join(IMAGES_DIR, "mask_blue.png"), mask_blue)
cv2.imwrite(os.path.join(IMAGES_DIR, "result_blue.png"), result_blue)
print("蓝色提取完成")

# === 提取黄色区域 ===
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
result_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)

cv2.imwrite(os.path.join(IMAGES_DIR, "mask_yellow.png"), mask_yellow)
cv2.imwrite(os.path.join(IMAGES_DIR, "result_yellow.png"), result_yellow)
print("黄色提取完成")

# 拼接对比图：原图 | 红 | 绿 | 蓝
comparison = np.hstack([img, result_red, result_green, result_blue])
cv2.imwrite(os.path.join(IMAGES_DIR, "color_filter_comparison.png"), comparison)
print(f"颜色过滤对比图已保存: {os.path.join(IMAGES_DIR, 'color_filter_comparison.png')}")
