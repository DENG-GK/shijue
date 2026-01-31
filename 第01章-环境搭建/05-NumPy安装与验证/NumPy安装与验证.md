# NumPy 安装与验证

> NumPy (Numerical Python) 是 Python 科学计算的基础库，提供高性能的多维数组对象和数学函数。

---

## 📖 理论部分

### 1. NumPy 简介

**NumPy** 是 Python 科学计算的核心库。

**核心特性：**
- ✅ **高效数组**：ndarray 多维数组，比 Python list 快 10-100 倍
- ✅ **向量化运算**：无需循环，直接对数组操作
- ✅ **数学函数**：线性代数、傅里叶变换、随机数等
- ✅ **广播机制**：自动处理不同形状数组的运算
- ✅ **C 语言实现**：底层用 C 编写，性能卓越

**为什么计算机视觉需要 NumPy？**
- 图像本质是数组（高度×宽度×通道）
- OpenCV 图像就是 NumPy 数组
- 高效的数值计算和数组操作
- 深度学习框架都基于 NumPy 或类似接口

---

### 2. NumPy 的重要性

在计算机视觉中，NumPy 无处不在：

```python
# OpenCV 读取的图像就是 NumPy 数组
img = cv2.imread('image.jpg')  # 返回 numpy.ndarray
print(type(img))  # <class 'numpy.ndarray'>
print(img.shape)  # (高度, 宽度, 通道数)
```

**NumPy 在 CV 中的应用：**
- 图像数组创建与操作
- 像素值计算与变换
- 矩阵运算（旋转、缩放）
- 数据归一化与标准化
- 批量处理图像

---

### 3. 安装方法

NumPy 通常作为 OpenCV 的依赖自动安装，也可以单独安装。

```bash
# pip 安装
pip install numpy

# 指定版本
pip install numpy==1.24.3

# conda 安装
conda install numpy
```

---

## 💻 代码实战

### pip 安装 NumPy

```bash
# 激活虚拟环境
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # macOS/Linux

# 安装 NumPy
pip install numpy

# 从国内镜像安装
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 验证安装

创建 `test_numpy.py`：

```python
# test_numpy.py
import numpy as np

print("=" * 60)
print("🎉 NumPy 安装验证")
print("=" * 60)

# 显示版本
print(f"\n📌 NumPy 版本：{np.__version__}")

# 显示配置信息
print(f"\n📦 NumPy 配置信息：")
np.show_config()

print("\n" + "=" * 60)
print("✅ NumPy 安装成功！")
print("=" * 60)
```

---

### 基础使用示例

创建 `numpy_basics.py`：

```python
# numpy_basics.py
import numpy as np

print("=" * 60)
print("NumPy 基础操作示例")
print("=" * 60)

# 1. 创建数组
print("\n1️⃣  创建数组")
arr1 = np.array([1, 2, 3, 4, 5])
print(f"一维数组：{arr1}")
print(f"形状：{arr1.shape}, 类型：{arr1.dtype}")

arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n二维数组：\n{arr2}")
print(f"形状：{arr2.shape}, 维度：{arr2.ndim}")

# 2. 特殊数组
print("\n2️⃣  特殊数组")
zeros = np.zeros((3, 4))
print(f"全零数组 (3x4)：\n{zeros}")

ones = np.ones((2, 3))
print(f"\n全一数组 (2x3)：\n{ones}")

eye = np.eye(3)
print(f"\n单位矩阵 (3x3)：\n{eye}")

random = np.random.rand(2, 3)
print(f"\n随机数组 (2x3)：\n{random}")

# 3. 数组运算
print("\n3️⃣  数组运算")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a ** 2 = {a ** 2}")

# 4. 索引与切片
print("\n4️⃣  索引与切片")
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print(f"原数组：\n{arr}")
print(f"第一行：{arr[0]}")
print(f"第二列：{arr[:, 1]}")
print(f"前两行前三列：\n{arr[:2, :3]}")

# 5. 形状操作
print("\n5️⃣  形状操作")
arr = np.arange(12)
print(f"一维数组：{arr}")

arr2d = arr.reshape(3, 4)
print(f"重塑为3x4：\n{arr2d}")

arr3d = arr.reshape(2, 2, 3)
print(f"重塑为2x2x3：\n{arr3d}")

# 6. 统计函数
print("\n6️⃣  统计函数")
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"数据：{data}")
print(f"总和：{np.sum(data)}")
print(f"平均值：{np.mean(data)}")
print(f"标准差：{np.std(data)}")
print(f"最大值：{np.max(data)}")
print(f"最小值：{np.min(data)}")

print("\n" + "=" * 60)
```

**运行：**
```bash
python numpy_basics.py
```

---

### NumPy 与图像处理

创建 `numpy_image.py`：

```python
# numpy_image.py
import numpy as np
import cv2

print("=" * 60)
print("NumPy 在图像处理中的应用")
print("=" * 60)

# 1. 用 NumPy 创建图像
print("\n1️⃣  创建图像数组")
# 创建 200x300 的黑色图像（3通道）
img = np.zeros((200, 300, 3), dtype=np.uint8)
print(f"图像形状：{img.shape}")
print(f"数据类型：{img.dtype}")
print(f"总像素数：{img.size}")

# 2. 像素操作
print("\n2️⃣  像素操作")
# 设置红色区域
img[50:150, 50:150] = [0, 0, 255]  # BGR: 红色
# 设置绿色区域
img[50:150, 150:250] = [0, 255, 0]  # BGR: 绿色

cv2.imwrite('numpy_image.png', img)
print("图像已保存：numpy_image.png")

# 3. 数组运算
print("\n3️⃣  图像运算")
# 创建两个图像
img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
img2 = np.ones((100, 100, 3), dtype=np.uint8) * 50

# 图像相加
result_add = cv2.add(img1, img2)
print(f"图像相加：{result_add[0, 0]}")

# 图像混合
result_blend = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
print(f"图像混合：{result_blend[0, 0]}")

# 4. 图像统计
print("\n4️⃣  图像统计")
img_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
print(f"平均亮度：{np.mean(img_gray):.2f}")
print(f"最大亮度：{np.max(img_gray)}")
print(f"最小亮度：{np.min(img_gray)}")
print(f"标准差：{np.std(img_gray):.2f}")

# 5. 图像归一化
print("\n5️⃣  图像归一化")
img_float = img_gray.astype(np.float32) / 255.0
print(f"归一化后范围：[{img_float.min():.3f}, {img_float.max():.3f}]")

print("\n" + "=" * 60)
```

---

## 🎯 总结

本节学习了 NumPy 的安装与基础使用：

✅ **了解 NumPy**：科学计算基础库，高性能数组
✅ **安装 NumPy**：`pip install numpy`
✅ **验证安装**：检查版本和配置
✅ **基础操作**：数组创建、运算、索引、形状操作
✅ **图像应用**：NumPy 在计算机视觉中的应用

**核心概念：**
- `ndarray`：多维数组
- `shape`：数组形状
- `dtype`：数据类型
- 向量化运算：高效的数组计算

**下一步：**
👉 [06 - 常用库安装](../06-常用库安装/README.md)

---

## 📚 参考资料

- [NumPy 官网](https://numpy.org/)
- [NumPy 中文文档](https://www.numpy.org.cn/)
- [NumPy 快速入门](https://numpy.org/doc/stable/user/quickstart.html)
