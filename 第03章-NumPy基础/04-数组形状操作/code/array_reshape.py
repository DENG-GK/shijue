# -*- coding: utf-8 -*-
"""
数组形状操作 - 示例代码
"""
import numpy as np

print("=" * 50)
print("NumPy 数组形状操作")
print("=" * 50)

# ========== 1. reshape ==========
print("\n【1. reshape 改变形状】")
a = np.arange(12)
print(f"原始: {a}, shape={a.shape}")
b = a.reshape(3, 4)
print(f"reshape(3,4):\n{b}")
c = a.reshape(2, 2, 3)
print(f"reshape(2,2,3):\n{c}")
# 使用 -1 自动推导
d = a.reshape(4, -1)
print(f"reshape(4,-1):\n{d}")

# 图像维度变换
img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
flat = img.reshape(-1, 3)  # 展平为像素列表
print(f"\n图像 {img.shape} -> 像素列表 {flat.shape}")

# ========== 2. 展平 ==========
print("\n【2. 展平操作】")
a = np.array([[1, 2, 3], [4, 5, 6]])
print(f"原始:\n{a}")
print(f"ravel(): {a.ravel()}")
print(f"flatten(): {a.flatten()}")

# ========== 3. 转置 ==========
print("\n【3. 转置】")
a = np.array([[1, 2, 3], [4, 5, 6]])
print(f"原始 {a.shape}:\n{a}")
print(f"转置 {a.T.shape}:\n{a.T}")

# 三维转置（通道顺序变换）
img = np.random.randint(0, 256, (2, 3, 3), dtype=np.uint8)
print(f"\n图像 shape: {img.shape} (H, W, C)")
transposed = img.transpose(2, 0, 1)
print(f"转置后 shape: {transposed.shape} (C, H, W)")

# ========== 4. 拼接 ==========
print("\n【4. 数组拼接】")
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(f"a:\n{a}")
print(f"b:\n{b}")

h = np.hstack((a, b))
print(f"水平拼接 hstack:\n{h}")

v = np.vstack((a, b))
print(f"垂直拼接 vstack:\n{v}")

# concatenate
c0 = np.concatenate((a, b), axis=0)
print(f"concatenate axis=0:\n{c0}")
c1 = np.concatenate((a, b), axis=1)
print(f"concatenate axis=1:\n{c1}")

# ========== 5. 分割 ==========
print("\n【5. 数组分割】")
a = np.arange(12).reshape(2, 6)
print(f"原始:\n{a}")

parts = np.hsplit(a, 3)
print(f"水平分割为3份:")
for i, p in enumerate(parts):
    print(f"  部分{i}: {p.tolist()}")

parts = np.hsplit(a, [2, 4])
print(f"按位置[2,4]分割:")
for i, p in enumerate(parts):
    print(f"  部分{i}: {p.tolist()}")

# ========== 6. 添加/删除维度 ==========
print("\n【6. 维度操作】")
a = np.array([1, 2, 3])
print(f"原始: shape={a.shape}")

# 添加维度
row = a[np.newaxis, :]
col = a[:, np.newaxis]
print(f"np.newaxis行: shape={row.shape}, {row}")
print(f"np.newaxis列: shape={col.shape}")

# expand_dims
expanded = np.expand_dims(a, axis=0)
print(f"expand_dims(axis=0): shape={expanded.shape}")

# squeeze 删除大小为1的维度
b = np.array([[[1, 2, 3]]])
print(f"\n原始: shape={b.shape}")
squeezed = b.squeeze()
print(f"squeeze(): shape={squeezed.shape}, {squeezed}")

print("\n" + "=" * 50)
print("示例代码运行完成！")
