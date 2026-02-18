"""
示例1：创建不同形状的结构元素
学习如何使用 OpenCV 创建不同形状的结构元素
"""

import cv2
import numpy as np

# ===================== 创建结构元素 =====================

# 创建 5×5 的矩形结构元素
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 创建 5×5 的十字形结构元素
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# 创建 5×5 的椭圆形结构元素
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ===================== 打印结构元素 =====================

print("=" * 50)
print("矩形结构元素 (MORPH_RECT):")
print("=" * 50)
print(rect_kernel)
print()

print("=" * 50)
print("十字形结构元素 (MORPH_CROSS):")
print("=" * 50)
print(cross_kernel)
print()

print("=" * 50)
print("椭圆形结构元素 (MORPH_ELLIPSE):")
print("=" * 50)
print(ellipse_kernel)

# ===================== 不同大小的结构元素 =====================

print("\n" + "=" * 50)
print("不同大小的矩形结构元素:")
print("=" * 50)

for size in [3, 5, 7]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    print(f"\n{size}×{size} 矩形核:")
    print(kernel)
