# -*- coding: utf-8 -*-
"""
图像保存 - 示例代码
"""
import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

print("=" * 50)
print("OpenCV 图像保存")
print("=" * 50)

# 创建测试图像
img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
cv2.putText(img, "Save Test", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

# 1. 保存为不同格式
formats = {
    "output.jpg": None,
    "output.png": None,
    "output.bmp": None,
}
for fname, params in formats.items():
    path = os.path.join(IMAGES_DIR, fname)
    cv2.imwrite(path, img, params)
    size = os.path.getsize(path)
    print(f"{fname}: {size/1024:.1f} KB")

# 2. JPEG不同质量
print("\nJPEG质量对比:")
for quality in [10, 50, 95]:
    path = os.path.join(IMAGES_DIR, f"quality_{quality}.jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    size = os.path.getsize(path)
    print(f"  质量{quality}: {size/1024:.1f} KB")

# 3. PNG不同压缩级别
print("\nPNG压缩对比:")
for level in [0, 5, 9]:
    path = os.path.join(IMAGES_DIR, f"compress_{level}.png")
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, level])
    size = os.path.getsize(path)
    print(f"  级别{level}: {size/1024:.1f} KB")

# 清理测试文件
for f in os.listdir(IMAGES_DIR):
    if f.startswith(("output", "quality_", "compress_")):
        os.remove(os.path.join(IMAGES_DIR, f))
print("\n临时文件已清理")
print("示例代码运行完成！")
