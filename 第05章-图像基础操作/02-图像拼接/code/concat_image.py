# -*- coding: utf-8 -*-
"""图像拼接示例"""
import cv2, numpy as np, os
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

a = np.full((200, 200, 3), [255, 0, 0], dtype=np.uint8)  # 蓝
b = np.full((200, 200, 3), [0, 255, 0], dtype=np.uint8)  # 绿
c = np.full((200, 200, 3), [0, 0, 255], dtype=np.uint8)  # 红

h = np.hstack((a, b, c))
v = np.vstack((a, b, c))
cv2.imwrite(os.path.join(IMAGES_DIR, "hstack.png"), h)
cv2.imwrite(os.path.join(IMAGES_DIR, "vstack.png"), v)
print(f"水平拼接: {h.shape}, 垂直拼接: {v.shape}")
