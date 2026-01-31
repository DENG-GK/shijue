# -*- coding: utf-8 -*-
"""
数组索引与切片 - 示例代码
"""
import numpy as np

print("=" * 50)
print("NumPy 数组索引与切片")
print("=" * 50)

# ========== 1. 一维数组索引 ==========
print("\n【1. 一维数组索引】")
a = np.array([10, 20, 30, 40, 50])
print(f"数组: {a}")
print(f"a[0] = {a[0]}, a[-1] = {a[-1]}")
print(f"a[1:4] = {a[1:4]}")
print(f"a[::2] = {a[::2]}")  # 每隔一个
print(f"a[::-1] = {a[::-1]}")  # 反转

# ========== 2. 二维数组索引（图像操作） ==========
print("\n【2. 二维数组索引】")
img = np.array([
    [0,   50, 100, 150, 200],
    [25,  75, 125, 175, 225],
    [50, 100, 150, 200, 250],
    [75, 125, 175, 225, 255]
], dtype=np.uint8)
print(f"模拟灰度图:\n{img}")
print(f"shape: {img.shape}")

# 单元素
print(f"\nimg[0, 0] = {img[0, 0]}")
print(f"img[2, 3] = {img[2, 3]}")
print(f"img[-1, -1] = {img[-1, -1]}")

# 行切片
print(f"\n第一行: {img[0, :]}")
print(f"最后一行: {img[-1, :]}")

# 列切片
print(f"第一列: {img[:, 0]}")
print(f"最后一列: {img[:, -1]}")

# ROI（感兴趣区域）
roi = img[1:3, 1:4]
print(f"\nROI img[1:3, 1:4]:\n{roi}")

# ========== 3. 三维数组索引（彩色图像） ==========
print("\n【3. 三维数组索引（彩色图像）】")
color_img = np.zeros((4, 5, 3), dtype=np.uint8)
color_img[:, :, 0] = 255  # B通道全255
color_img[:, :, 1] = 128  # G通道全128
color_img[:, :, 2] = 0    # R通道全0
print(f"彩色图shape: {color_img.shape}")
print(f"像素(0,0)的BGR: {color_img[0, 0]}")

# 提取单通道
b_channel = color_img[:, :, 0]
g_channel = color_img[:, :, 1]
r_channel = color_img[:, :, 2]
print(f"B通道均值: {b_channel.mean()}")
print(f"G通道均值: {g_channel.mean()}")
print(f"R通道均值: {r_channel.mean()}")

# ========== 4. 布尔索引 ==========
print("\n【4. 布尔索引（条件筛选）】")
a = np.array([10, 150, 30, 200, 50, 255, 0, 128])
print(f"数组: {a}")

# 条件筛选
mask = a > 100
print(f"a > 100: {mask}")
print(f"大于100的元素: {a[mask]}")

# 阈值处理模拟
threshold = 127
binary = np.where(a > threshold, 255, 0).astype(np.uint8)
print(f"阈值{threshold}二值化: {binary}")

# 范围筛选
in_range = a[(a >= 50) & (a <= 200)]
print(f"50-200范围内: {in_range}")

# 修改满足条件的元素
b = a.copy()
b[b > 200] = 200  # 裁剪到200
print(f"裁剪到200: {b}")

# ========== 5. 花式索引 ==========
print("\n【5. 花式索引】")
a = np.array([10, 20, 30, 40, 50, 60, 70])
indices = [0, 2, 5]
print(f"数组: {a}")
print(f"a[{indices}] = {a[indices]}")

# 二维花式索引
rows = [0, 1, 2]
cols = [1, 3, 4]
print(f"\nimg花式索引:")
print(f"img[{rows}, {cols}] = {img[rows, cols]}")

# ========== 6. 图像处理实战 ==========
print("\n【6. 图像处理实战】")

# 创建渐变图像
gradient = np.tile(np.arange(0, 256, 25.6, dtype=np.uint8), (8, 1))
print(f"渐变图像:\n{gradient}")

# 提取ROI
roi = gradient[2:6, 3:7]
print(f"\nROI:\n{roi}")

# 像素替换
result = gradient.copy()
result[result < 100] = 0
result[result >= 100] = 255
print(f"\n二值化:\n{result}")

print("\n" + "=" * 50)
print("示例代码运行完成！")
