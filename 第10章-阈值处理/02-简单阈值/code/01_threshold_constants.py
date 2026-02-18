"""
查看所有阈值类型常量
"""

import cv2

# 阈值类型及其数值
threshold_types = {
    'THRESH_BINARY': cv2.THRESH_BINARY,
    'THRESH_BINARY_INV': cv2.THRESH_BINARY_INV,
    'THRESH_TRUNC': cv2.THRESH_TRUNC,
    'THRESH_TOZERO': cv2.THRESH_TOZERO,
    'THRESH_TOZERO_INV': cv2.THRESH_TOZERO_INV,
    'THRESH_OTSU': cv2.THRESH_OTSU,
    'THRESH_TRIANGLE': cv2.THRESH_TRIANGLE,
}

print("OpenCV 阈值类型常量：")
print("-" * 40)
for name, value in threshold_types.items():
    print(f"{name:20s} = {value}")
print("-" * 40)
print("\n提示：OTSU和TRIANGLE可以与其他类型组合使用")
print("例如：cv2.THRESH_BINARY + cv2.THRESH_OTSU")
