# -*- coding: utf-8 -*-
"""
NumPy 常用函数 - 示例代码
"""
import numpy as np
import os

print("=" * 50)
print("NumPy 常用函数")
print("=" * 50)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 1. 数学函数 ==========
print("\n【1. 数学函数】")
a = np.array([0, 30, 45, 60, 90])
radians = np.radians(a)
print(f"角度: {a}")
print(f"sin: {np.sin(radians).round(4)}")
print(f"cos: {np.cos(radians).round(4)}")

b = np.array([1, 2, 4, 8, 16])
print(f"\n数组: {b}")
print(f"sqrt: {np.sqrt(b).round(4)}")
print(f"log2: {np.log2(b).round(4)}")
print(f"exp: {np.exp([0, 1, 2]).round(4)}")

# 图像处理中的应用：伽马校正
pixels = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
gamma = 2.2
corrected = (255 * np.power(pixels / 255.0, 1/gamma)).astype(np.uint8)
print(f"\n伽马校正 (gamma={gamma}):")
print(f"  原始: {pixels}")
print(f"  校正: {corrected}")

# ========== 2. 排序 ==========
print("\n【2. 排序】")
a = np.array([50, 10, 40, 20, 30])
print(f"原始: {a}")
print(f"sort: {np.sort(a)}")
print(f"argsort: {np.argsort(a)}")  # 排序后的索引

# 按行/列排序
matrix = np.array([[3, 1, 2], [6, 4, 5]])
print(f"\n矩阵:\n{matrix}")
print(f"按行排序:\n{np.sort(matrix, axis=1)}")
print(f"按列排序:\n{np.sort(matrix, axis=0)}")

# ========== 3. 查找 ==========
print("\n【3. 查找函数】")
a = np.array([10, 50, 200, 30, 255, 0, 128])
print(f"数组: {a}")
print(f"最大值索引: {np.argmax(a)} -> {a[np.argmax(a)]}")
print(f"最小值索引: {np.argmin(a)} -> {a[np.argmin(a)]}")

# np.where
result = np.where(a > 100, 255, 0)
print(f"where(a>100, 255, 0): {result}")

# 非零元素
indices = np.nonzero(a)
print(f"非零索引: {indices[0]}")

# ========== 4. unique ==========
print("\n【4. unique 去重】")
labels = np.array([0, 1, 1, 2, 0, 2, 3, 1, 3])
unique_vals, counts = np.unique(labels, return_counts=True)
print(f"标签: {labels}")
print(f"唯一值: {unique_vals}")
print(f"计数: {counts}")

# ========== 5. 文件IO ==========
print("\n【5. 文件IO】")
data = np.random.randint(0, 256, (3, 4), dtype=np.uint8)
print(f"原始数据:\n{data}")

# 二进制格式
npy_file = os.path.join(SCRIPT_DIR, "test_data.npy")
np.save(npy_file, data)
loaded = np.load(npy_file)
print(f"npy加载:\n{loaded}")
print(f"数据一致: {np.array_equal(data, loaded)}")

# 文本格式
txt_file = os.path.join(SCRIPT_DIR, "test_data.txt")
np.savetxt(txt_file, data, fmt='%d', delimiter=',')
loaded_txt = np.loadtxt(txt_file, delimiter=',', dtype=np.uint8)
print(f"txt加载:\n{loaded_txt}")

# 清理
os.remove(npy_file)
os.remove(txt_file)
print("临时文件已清理")

# ========== 6. 实用工具 ==========
print("\n【6. 实用工具】")

# clip 裁剪
a = np.array([-10, 50, 150, 300])
clipped = np.clip(a, 0, 255)
print(f"clip({a}, 0, 255): {clipped}")

# copy 深拷贝
a = np.array([1, 2, 3])
b = a.copy()
b[0] = 100
print(f"\n深拷贝: a={a}, b={b}")

# 视图（浅拷贝）
c = a.view()
c[0] = 999
print(f"视图: a={a}, c={c}")

# linspace 在图像处理中创建查找表
lut = np.linspace(0, 255, 256, dtype=np.uint8)
print(f"\n查找表(前10个): {lut[:10]}")

# meshgrid 创建坐标网格
x = np.arange(3)
y = np.arange(4)
xx, yy = np.meshgrid(x, y)
print(f"\nmeshgrid:")
print(f"xx:\n{xx}")
print(f"yy:\n{yy}")

print("\n" + "=" * 50)
print("示例代码运行完成！")
