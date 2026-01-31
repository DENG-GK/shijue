# -*- coding: utf-8 -*-
"""
数组运算 - 示例代码
"""
import numpy as np

print("=" * 50)
print("NumPy 数组运算")
print("=" * 50)

# ========== 1. 算术运算 ==========
print("\n【1. 算术运算（逐元素）】")
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])
print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a // b = {a // b}")
print(f"a ** 2 = {a ** 2}")

# 图像亮度调节
print("\n图像亮度调节:")
pixels = np.array([100, 150, 200, 50], dtype=np.uint8)
brighter = np.clip(pixels.astype(np.int16) + 50, 0, 255).astype(np.uint8)
darker = np.clip(pixels.astype(np.int16) - 50, 0, 255).astype(np.uint8)
print(f"原始: {pixels}")
print(f"加亮+50: {brighter}")
print(f"变暗-50: {darker}")

# ========== 2. 广播机制 ==========
print("\n【2. 广播机制】")

# 标量广播
a = np.array([[1, 2, 3], [4, 5, 6]])
print(f"数组:\n{a}")
print(f"数组 * 10:\n{a * 10}")

# 图像归一化（所有像素除以255）
img = np.array([[0, 128, 255], [64, 192, 32]], dtype=np.uint8)
normalized = img.astype(np.float32) / 255.0
print(f"\n归一化前:\n{img}")
print(f"归一化后:\n{normalized}")

# 一维与二维广播
row = np.array([10, 20, 30])
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n矩阵:\n{matrix}")
print(f"行向量: {row}")
print(f"矩阵 + 行向量:\n{matrix + row}")

# ========== 3. 矩阵运算 ==========
print("\n【3. 矩阵运算】")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"A:\n{A}")
print(f"B:\n{B}")
print(f"A @ B:\n{A @ B}")
print(f"np.dot(A, B):\n{np.dot(A, B)}")

# 转置
print(f"\nA.T:\n{A.T}")

# ========== 4. 统计运算 ==========
print("\n【4. 统计运算】")
img = np.random.randint(0, 256, (4, 5), dtype=np.uint8)
print(f"模拟图像:\n{img}")
print(f"总和: {img.sum()}")
print(f"均值: {img.mean():.2f}")
print(f"标准差: {img.std():.2f}")
print(f"最小值: {img.min()}, 位置: {np.unravel_index(img.argmin(), img.shape)}")
print(f"最大值: {img.max()}, 位置: {np.unravel_index(img.argmax(), img.shape)}")

# 按轴统计
print(f"\n每行均值: {img.mean(axis=1)}")
print(f"每列均值: {img.mean(axis=0)}")

# ========== 5. 比较运算 ==========
print("\n【5. 比较运算】")
a = np.array([10, 50, 100, 200, 255])
print(f"a = {a}")
print(f"a > 100: {a > 100}")
print(f"a == 100: {a == 100}")
print(f"满足 a>100 的个数: {np.sum(a > 100)}")
print(f"所有元素>0: {np.all(a > 0)}")
print(f"存在元素>200: {np.any(a > 200)}")

# np.where
result = np.where(a > 100, 255, 0)
print(f"np.where(a>100, 255, 0): {result}")

# np.clip
clipped = np.clip(a, 50, 200)
print(f"np.clip(a, 50, 200): {clipped}")

print("\n" + "=" * 50)
print("示例代码运行完成！")
