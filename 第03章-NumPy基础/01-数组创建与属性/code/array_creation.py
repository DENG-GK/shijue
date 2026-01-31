# -*- coding: utf-8 -*-
"""
数组创建与属性 - 示例代码
演示NumPy数组的创建方法和基本属性
"""
import numpy as np

print("=" * 50)
print("NumPy 数组创建与属性")
print("=" * 50)

# ========== 1. 从列表创建 ==========
print("\n【1. 从列表创建数组】")
a = np.array([1, 2, 3, 4, 5])
print(f"一维数组: {a}")

b = np.array([[1, 2, 3], [4, 5, 6]])
print(f"二维数组:\n{b}")

# 指定数据类型
c = np.array([1, 2, 3], dtype=np.float32)
print(f"float32数组: {c}, dtype={c.dtype}")

# 模拟灰度图像（uint8类型）
gray_img = np.array([[0, 64, 128], [192, 255, 32]], dtype=np.uint8)
print(f"模拟灰度图:\n{gray_img}")

# ========== 2. 特殊数组 ==========
print("\n【2. 特殊数组】")

# 全零数组（黑色图像）
zeros = np.zeros((3, 4), dtype=np.uint8)
print(f"全零数组（黑色图像）:\n{zeros}")

# 全一数组
ones = np.ones((2, 3), dtype=np.float64)
print(f"全一数组:\n{ones}")

# 全填充数组（灰色图像）
gray = np.full((3, 3), 128, dtype=np.uint8)
print(f"全128数组（灰色图像）:\n{gray}")

# 单位矩阵
eye = np.eye(3)
print(f"单位矩阵:\n{eye}")

# 空数组（未初始化，值随机）
empty = np.empty((2, 2))
print(f"空数组:\n{empty}")

# ========== 3. 序列数组 ==========
print("\n【3. 序列数组】")

# arange（类似range）
arr1 = np.arange(0, 10, 2)
print(f"arange(0, 10, 2): {arr1}")

# linspace（等间距）
arr2 = np.linspace(0, 1, 5)
print(f"linspace(0, 1, 5): {arr2}")

# 像素值范围
pixel_range = np.arange(0, 256, 51)
print(f"像素值采样: {pixel_range}")

# logspace（对数等间距）
arr3 = np.logspace(0, 3, 4)
print(f"logspace(0, 3, 4): {arr3}")

# ========== 4. 随机数组 ==========
print("\n【4. 随机数组】")

np.random.seed(42)  # 设置随机种子保证可重复

# 均匀分布 [0, 1)
rand_uniform = np.random.rand(2, 3)
print(f"均匀分布:\n{rand_uniform}")

# 正态分布
rand_normal = np.random.randn(2, 3)
print(f"正态分布:\n{rand_normal}")

# 整数随机（模拟随机像素值）
rand_int = np.random.randint(0, 256, size=(3, 4), dtype=np.uint8)
print(f"随机像素值:\n{rand_int}")

# 随机选择
choices = np.random.choice([0, 128, 255], size=(2, 3))
print(f"随机选择:\n{choices}")

# ========== 5. 模拟图像创建 ==========
print("\n【5. 模拟图像创建】")

# 模拟 480x640 灰度图
gray_image = np.zeros((480, 640), dtype=np.uint8)
print(f"灰度图形状: {gray_image.shape}")

# 模拟 480x640 BGR彩色图
color_image = np.zeros((480, 640, 3), dtype=np.uint8)
print(f"彩色图形状: {color_image.shape}")

# 创建纯色图像
red_image = np.zeros((100, 100, 3), dtype=np.uint8)
red_image[:, :, 2] = 255  # BGR中R通道在索引2
print(f"红色图像: shape={red_image.shape}, R通道均值={red_image[:,:,2].mean()}")

# ========== 6. 数组属性 ==========
print("\n【6. 数组属性】")

img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
print(f"模拟图像属性:")
print(f"  ndim (维度数): {img.ndim}")
print(f"  shape (形状): {img.shape}")
print(f"  size (元素总数): {img.size}")
print(f"  dtype (数据类型): {img.dtype}")
print(f"  itemsize (字节/元素): {img.itemsize}")
print(f"  nbytes (总字节数): {img.nbytes}")
print(f"  总大小: {img.nbytes / 1024:.1f} KB")

# 不同图像类型的对比
print("\n不同图像类型对比:")
types = [
    ("灰度图 uint8", np.zeros((480, 640), dtype=np.uint8)),
    ("彩色图 uint8", np.zeros((480, 640, 3), dtype=np.uint8)),
    ("彩色图 float32", np.zeros((480, 640, 3), dtype=np.float32)),
    ("彩色图 float64", np.zeros((480, 640, 3), dtype=np.float64)),
]
for name, arr in types:
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes/1024:.0f}KB")

# ========== 7. 数据类型转换 ==========
print("\n【7. 数据类型转换】")

# uint8 → float32（常用于归一化）
uint8_arr = np.array([0, 128, 255], dtype=np.uint8)
float32_arr = uint8_arr.astype(np.float32) / 255.0
print(f"uint8: {uint8_arr} -> float32: {float32_arr}")

# float32 → uint8（显示图像时）
back_to_uint8 = (float32_arr * 255).astype(np.uint8)
print(f"float32: {float32_arr} -> uint8: {back_to_uint8}")

print("\n" + "=" * 50)
print("示例代码运行完成！")
