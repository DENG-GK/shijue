# -*- coding: utf-8 -*-
"""通道分离与合并示例"""
import cv2
import numpy as np
import os

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
SRC_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "01-颜色空间理论", "images", "color_shapes.png")

if not os.path.exists(SRC_IMG):
    print("请先运行 01-颜色空间理论/code/generate_images.py 生成测试图片")
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(img, (150, 200), 80, (255, 0, 0), -1)
    cv2.circle(img, (300, 200), 80, (0, 255, 0), -1)
    cv2.circle(img, (450, 200), 80, (0, 0, 255), -1)
else:
    img = cv2.imread(SRC_IMG)

os.makedirs(IMAGES_DIR, exist_ok=True)

# === 通道分离 ===
# 方式1：cv2.split()
b, g, r = cv2.split(img)
print(f"原图 shape: {img.shape}")
print(f"单通道 shape: {b.shape}")

# 方式2：数组索引（更高效）
b2 = img[:, :, 0]
g2 = img[:, :, 1]
r2 = img[:, :, 2]
print(f"索引方式结果一致: {np.array_equal(b, b2)}")

# 保存单通道灰度图
cv2.imwrite(os.path.join(IMAGES_DIR, "channel_b.png"), b)
cv2.imwrite(os.path.join(IMAGES_DIR, "channel_g.png"), g)
cv2.imwrite(os.path.join(IMAGES_DIR, "channel_r.png"), r)

# === 彩色可视化单通道 ===
zeros = np.zeros_like(b)
blue_only = cv2.merge([b, zeros, zeros])
green_only = cv2.merge([zeros, g, zeros])
red_only = cv2.merge([zeros, zeros, r])

cv2.imwrite(os.path.join(IMAGES_DIR, "blue_only.png"), blue_only)
cv2.imwrite(os.path.join(IMAGES_DIR, "green_only.png"), green_only)
cv2.imwrite(os.path.join(IMAGES_DIR, "red_only.png"), red_only)

# === 通道合并 ===
merged = cv2.merge([b, g, r])
print(f"合并后 shape: {merged.shape}")
print(f"与原图一致: {np.array_equal(img, merged)}")

# 拼接对比图
comparison = np.hstack([img, blue_only, green_only, red_only])
cv2.imwrite(os.path.join(IMAGES_DIR, "channel_comparison.png"), comparison)
print(f"对比图已保存: {os.path.join(IMAGES_DIR, 'channel_comparison.png')}")
