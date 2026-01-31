# -*- coding: utf-8 -*-
"""
函数与模块 - 示例代码
本脚本演示Python函数定义和模块使用
"""

print("=" * 50)
print("Python 函数与模块示例")
print("=" * 50)

# ========== 1. 基本函数定义 ==========
print("\n【1. 基本函数定义】")

def greet(name):
    """简单的问候函数"""
    return f"你好, {name}!"

result = greet("Python")
print(result)

# 无返回值函数
def print_separator():
    """打印分隔线"""
    print("-" * 30)

print_separator()

# 多返回值
def get_image_info():
    """返回图像信息（多个值）"""
    width = 640
    height = 480
    channels = 3
    return width, height, channels

w, h, c = get_image_info()
print(f"图像信息: {w}x{h}, {c}通道")

# ========== 2. 参数类型 ==========
print("\n【2. 参数类型】")

# 位置参数
def add(a, b):
    """位置参数示例"""
    return a + b

print(f"add(3, 5) = {add(3, 5)}")

# 默认参数
def resize_image(width, height, scale=1.0):
    """带默认参数的函数"""
    new_width = int(width * scale)
    new_height = int(height * scale)
    return new_width, new_height

print(f"resize_image(640, 480) = {resize_image(640, 480)}")
print(f"resize_image(640, 480, 0.5) = {resize_image(640, 480, 0.5)}")

# 关键字参数
def create_image(width, height, channels=3, fill=0):
    """使用关键字参数"""
    return f"创建 {width}x{height}x{channels} 图像, 填充值={fill}"

print(create_image(640, 480))
print(create_image(640, 480, fill=255))
print(create_image(width=800, height=600, channels=1))

# *args - 可变位置参数
def calculate_mean(*args):
    """计算任意数量参数的平均值"""
    if len(args) == 0:
        return 0
    return sum(args) / len(args)

print(f"calculate_mean(1, 2, 3) = {calculate_mean(1, 2, 3)}")
print(f"calculate_mean(10, 20, 30, 40, 50) = {calculate_mean(10, 20, 30, 40, 50)}")

# **kwargs - 可变关键字参数
def print_config(**kwargs):
    """打印配置信息"""
    print("配置信息:")
    for key, value in kwargs.items():
        print(f"  {key} = {value}")

print_config(width=640, height=480, fps=30, format="RGB")

# 混合参数
def process_image(image_path, *filters, output=None, **options):
    """混合参数示例"""
    print(f"处理图像: {image_path}")
    print(f"应用滤镜: {filters}")
    print(f"输出路径: {output}")
    print(f"其他选项: {options}")

print("\n混合参数示例:")
process_image("test.jpg", "blur", "sharpen", output="result.jpg", quality=95)

# ========== 3. Lambda 表达式 ==========
print("\n【3. Lambda 表达式】")

# 基本 lambda
square = lambda x: x ** 2
print(f"square(5) = {square(5)}")

# 多参数 lambda
add = lambda x, y: x + y
print(f"add(3, 4) = {add(3, 4)}")

# 在排序中使用 lambda
points = [(1, 5), (3, 2), (2, 8), (4, 1)]
print(f"原始点: {points}")

# 按 y 坐标排序
sorted_by_y = sorted(points, key=lambda p: p[1])
print(f"按y排序: {sorted_by_y}")

# 按距离原点排序
sorted_by_dist = sorted(points, key=lambda p: p[0]**2 + p[1]**2)
print(f"按距离排序: {sorted_by_dist}")

# 在 map/filter 中使用
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(f"map平方: {squared}")

evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"filter偶数: {evens}")

# ========== 4. 函数作为参数 ==========
print("\n【4. 函数作为参数】")

def apply_to_image(image_data, transform_func):
    """将变换函数应用到图像数据"""
    return [transform_func(pixel) for pixel in image_data]

pixels = [50, 100, 150, 200]

# 使用普通函数
def invert(x):
    return 255 - x

inverted = apply_to_image(pixels, invert)
print(f"反色: {pixels} -> {inverted}")

# 使用 lambda
doubled = apply_to_image(pixels, lambda x: min(255, x * 2))
print(f"加倍: {pixels} -> {doubled}")

# ========== 5. 闭包 ==========
print("\n【5. 闭包】")

def create_threshold_func(threshold):
    """创建阈值函数的工厂"""
    def threshold_func(value):
        return 255 if value > threshold else 0
    return threshold_func

# 创建不同阈值的函数
thresh_100 = create_threshold_func(100)
thresh_200 = create_threshold_func(200)

pixel = 150
print(f"像素 {pixel}:")
print(f"  阈值100: {thresh_100(pixel)}")
print(f"  阈值200: {thresh_200(pixel)}")

# ========== 6. 装饰器 ==========
print("\n【6. 装饰器】")

def timer_decorator(func):
    """计时装饰器"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"  {func.__name__} 耗时: {(end-start)*1000:.2f}ms")
        return result
    return wrapper

@timer_decorator
def slow_function(n):
    """模拟耗时操作"""
    total = 0
    for i in range(n):
        total += i
    return total

result = slow_function(100000)
print(f"  结果: {result}")

# ========== 7. 模块导入 ==========
print("\n【7. 模块导入】")

# 导入标准库模块
import math
print(f"math.pi = {math.pi}")
print(f"math.sqrt(16) = {math.sqrt(16)}")

# 从模块导入特定函数
from math import sin, cos, radians
angle = 45
print(f"sin({angle}°) = {sin(radians(angle)):.4f}")
print(f"cos({angle}°) = {cos(radians(angle)):.4f}")

# 使用别名
import os.path as path
print(f"当前文件存在: {path.exists(__file__)}")

# 常用模块示例
import random
print(f"随机整数 1-10: {random.randint(1, 10)}")
print(f"随机选择: {random.choice(['A', 'B', 'C'])}")

import datetime
now = datetime.datetime.now()
print(f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")

# ========== 8. 文档字符串 ==========
print("\n【8. 文档字符串】")

def normalize_pixel(value, min_val=0, max_val=255):
    """
    将像素值归一化到 [0, 1] 范围。

    参数:
        value: 原始像素值
        min_val: 最小值（默认0）
        max_val: 最大值（默认255）

    返回:
        归一化后的值（0到1之间的浮点数）

    示例:
        >>> normalize_pixel(128)
        0.5019607843137255
    """
    return (value - min_val) / (max_val - min_val)

# 查看文档
print(f"函数文档:\n{normalize_pixel.__doc__}")
print(f"normalize_pixel(128) = {normalize_pixel(128):.4f}")

print("\n" + "=" * 50)
print("示例代码运行完成！")
