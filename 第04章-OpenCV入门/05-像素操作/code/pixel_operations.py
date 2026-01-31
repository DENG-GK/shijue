# -*- coding: utf-8 -*-
"""
像素操作 - 示例代码
"""
import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

print("=" * 50)
print("OpenCV 像素操作")
print("=" * 50)

# 创建测试图像
img = np.zeros((200, 300, 3), dtype=np.uint8)
img[:] = [100, 150, 200]  # BGR

# 1. 像素访问
print("\n【1. 像素访问】")
print(f"像素(0,0) BGR: {img[0, 0]}")
print(f"B通道(0,0): {img[0, 0, 0]}")

# 修改区域
img[50:150, 100:200] = [0, 255, 0]  # 绿色区域
print("已修改区域像素")

# 2. 通道操作
print("\n【2. 通道拆分与合并】")
b, g, r = cv2.split(img)
print(f"B通道: shape={b.shape}, mean={b.mean():.1f}")
print(f"G通道: shape={g.shape}, mean={g.mean():.1f}")
print(f"R通道: shape={r.shape}, mean={r.mean():.1f}")

merged = cv2.merge([b, g, r])
print(f"合并后: shape={merged.shape}")

# 3. 算术运算
print("\n【3. 图像算术运算】")
a = np.full((100, 100, 3), 200, dtype=np.uint8)
b = np.full((100, 100, 3), 100, dtype=np.uint8)

# cv2.add 饱和运算 vs NumPy取模运算
cv_add = cv2.add(a, b)
np_add = a + b
print(f"cv2.add(200, 100) = {cv_add[0,0,0]}")   # 255 (饱和)
print(f"numpy加(200+100) = {np_add[0,0,0]}")      # 44  (取模)

# 加权融合
img1 = np.full((200, 300, 3), [255, 0, 0], dtype=np.uint8)
img2 = np.full((200, 300, 3), [0, 0, 255], dtype=np.uint8)
blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
print(f"加权融合: {blended[0,0]}")

# 保存结果
cv2.imwrite(os.path.join(IMAGES_DIR, "blended.png"), blended)
print("\n示例代码运行完成！")
