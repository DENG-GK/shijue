# -*- coding: utf-8 -*-
"""图像旋转示例"""
import cv2, numpy as np, os
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.putText(img, "Rotate", (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

# 简单旋转
r90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
r180 = cv2.rotate(img, cv2.ROTATE_180)

# 任意角度旋转
h, w = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
r45 = cv2.warpAffine(img, M, (w, h))

# 翻转
flip_h = cv2.flip(img, 1)  # 水平
flip_v = cv2.flip(img, 0)  # 垂直

for name, result in [("r90", r90), ("r180", r180), ("r45", r45), ("flip_h", flip_h), ("flip_v", flip_v)]:
    cv2.imwrite(os.path.join(IMAGES_DIR, f"{name}.png"), result)
    print(f"{name}: {result.shape}")

print("旋转结果已保存")
