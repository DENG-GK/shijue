# -*- coding: utf-8 -*-
"""
图像属性 - 示例代码
"""
import cv2
import numpy as np

print("=" * 50)
print("OpenCV 图像属性")
print("=" * 50)

# 模拟不同类型图像
images = {
    "灰度图 uint8": np.zeros((480, 640), dtype=np.uint8),
    "彩色图 uint8": np.zeros((480, 640, 3), dtype=np.uint8),
    "RGBA图 uint8": np.zeros((480, 640, 4), dtype=np.uint8),
    "彩色图 float32": np.zeros((480, 640, 3), dtype=np.float32),
}

for name, img in images.items():
    print(f"\n{name}:")
    print(f"  shape: {img.shape}")
    print(f"  ndim: {img.ndim}")
    print(f"  size: {img.size}")
    print(f"  dtype: {img.dtype}")
    print(f"  内存: {img.nbytes / 1024:.0f} KB")

    # 获取宽高
    if img.ndim == 2:
        h, w = img.shape
        channels = 1
    else:
        h, w, channels = img.shape
    print(f"  宽: {w}, 高: {h}, 通道: {channels}")

print("\n示例代码运行完成！")
