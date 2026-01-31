# -*- coding: utf-8 -*-
"""
控制流程 - 示例代码
本脚本演示Python的条件判断和循环结构
"""

print("=" * 50)
print("Python 控制流程示例")
print("=" * 50)

# ========== 1. 条件判断 ==========
print("\n【1. 条件判断】")

# 基本 if-else
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"
print(f"分数 {score} 对应等级: {grade}")

# 图像处理中的条件判断示例
pixel_value = 128
if pixel_value < 50:
    category = "暗区域"
elif pixel_value < 200:
    category = "中等亮度"
else:
    category = "亮区域"
print(f"像素值 {pixel_value} 属于: {category}")

# 逻辑运算符
width, height = 640, 480
if width > 0 and height > 0:
    print(f"有效图像尺寸: {width}x{height}")

is_color = True
has_alpha = False
if is_color or has_alpha:
    print("图像有颜色信息或透明通道")

if not has_alpha:
    print("图像没有透明通道")

# 三元表达式
threshold = 127
result = "二值化为白" if pixel_value > threshold else "二值化为黑"
print(f"像素 {pixel_value} {result}")

# ========== 2. for 循环 ==========
print("\n【2. for 循环】")

# 遍历列表
colors = ["红", "绿", "蓝"]
print("颜色通道:")
for color in colors:
    print(f"  - {color}")

# 使用 range
print("\n使用 range(5):")
for i in range(5):
    print(f"  i = {i}")

print("\n使用 range(2, 8, 2):")  # 起始、结束、步长
for i in range(2, 8, 2):
    print(f"  i = {i}")

# enumerate 带索引遍历
print("\n使用 enumerate:")
pixels = [255, 128, 64, 32]
for index, value in enumerate(pixels):
    print(f"  像素[{index}] = {value}")

# 遍历字典
print("\n遍历字典:")
image_info = {"width": 640, "height": 480, "channels": 3}
for key, value in image_info.items():
    print(f"  {key}: {value}")

# zip 同时遍历多个序列
print("\n使用 zip:")
names = ["R", "G", "B"]
values = [255, 128, 0]
for name, value in zip(names, values):
    print(f"  {name} = {value}")

# 嵌套循环（模拟图像遍历）
print("\n嵌套循环（3x3网格）:")
for row in range(3):
    for col in range(3):
        print(f"  ({row}, {col})", end="")
    print()  # 换行

# ========== 3. while 循环 ==========
print("\n【3. while 循环】")

# 基本 while
count = 0
print("while 循环计数:")
while count < 5:
    print(f"  count = {count}")
    count += 1

# 模拟迭代收敛
print("\n模拟迭代收敛:")
value = 100.0
iteration = 0
while value > 1.0 and iteration < 10:
    value = value / 2
    iteration += 1
    print(f"  迭代 {iteration}: value = {value:.2f}")

# ========== 4. 循环控制 ==========
print("\n【4. 循环控制】")

# break - 退出循环
print("break 示例（找到第一个偶数）:")
numbers = [1, 3, 5, 4, 7, 9]
for num in numbers:
    if num % 2 == 0:
        print(f"  找到偶数: {num}")
        break
    print(f"  检查: {num} 是奇数")

# continue - 跳过本次
print("\ncontinue 示例（跳过奇数）:")
for num in range(6):
    if num % 2 == 1:
        continue  # 跳过奇数
    print(f"  偶数: {num}")

# pass - 占位符
print("\npass 示例:")
for i in range(3):
    if i == 1:
        pass  # 暂时不处理
    else:
        print(f"  处理 i = {i}")

# for-else 结构
print("\nfor-else 示例:")
target = 10
numbers = [1, 3, 5, 7, 9]
for num in numbers:
    if num == target:
        print(f"  找到目标 {target}")
        break
else:
    print(f"  未找到目标 {target}")

# ========== 5. 列表推导式 ==========
print("\n【5. 列表推导式】")

# 基本列表推导式
squares = [x**2 for x in range(6)]
print(f"平方数: {squares}")

# 带条件的列表推导式
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"偶数的平方: {even_squares}")

# 字符串处理
words = ["hello", "world", "python"]
upper_words = [w.upper() for w in words]
print(f"大写: {upper_words}")

# 嵌套列表推导式（生成坐标）
coords = [(x, y) for x in range(3) for y in range(3)]
print(f"坐标网格: {coords}")

# 图像处理示例：像素归一化
pixels = [0, 64, 128, 192, 255]
normalized = [p / 255.0 for p in pixels]
print(f"原始像素: {pixels}")
print(f"归一化后: {[f'{n:.2f}' for n in normalized]}")

# 条件表达式在列表推导式中
values = [10, -5, 20, -15, 30]
abs_values = [v if v >= 0 else -v for v in values]
print(f"取绝对值: {abs_values}")

# 字典推导式
names = ["a", "b", "c"]
values = [1, 2, 3]
d = {k: v for k, v in zip(names, values)}
print(f"字典推导式: {d}")

# 集合推导式
nums = [1, 2, 2, 3, 3, 3, 4]
unique_squares = {x**2 for x in nums}
print(f"集合推导式（去重）: {unique_squares}")

# ========== 6. 实际应用示例 ==========
print("\n【6. 实际应用示例】")

# 模拟图像阈值处理
print("模拟阈值处理:")
image_row = [50, 100, 150, 200, 250]
threshold = 127
binary_row = [255 if p > threshold else 0 for p in image_row]
print(f"  原始: {image_row}")
print(f"  阈值 {threshold} 后: {binary_row}")

# 模拟查找最大值位置
print("\n查找最大值位置:")
pixels = [45, 120, 89, 200, 156, 78]
max_val = 0
max_idx = 0
for i, p in enumerate(pixels):
    if p > max_val:
        max_val = p
        max_idx = i
print(f"  像素值: {pixels}")
print(f"  最大值 {max_val} 在位置 {max_idx}")

# 使用内置函数更简洁
max_val = max(pixels)
max_idx = pixels.index(max_val)
print(f"  使用内置函数: 最大值 {max_val} 在位置 {max_idx}")

print("\n" + "=" * 50)
print("示例代码运行完成！")
