# -*- coding: utf-8 -*-
"""
数据类型与变量 - 示例代码
本脚本演示Python基本数据类型和变量操作
"""

print("=" * 50)
print("Python 数据类型与变量示例")
print("=" * 50)

# ========== 1. 基本数据类型 ==========
print("\n【1. 基本数据类型】")

# 整数 (int)
width = 640
height = 480
total_pixels = width * height
print(f"图像尺寸: {width}x{height}")
print(f"总像素数: {total_pixels}")
print(f"类型: {type(width)}")

# 浮点数 (float)
fps = 30.0
duration = 10.5
total_frames = fps * duration
print(f"\n帧率: {fps} fps")
print(f"时长: {duration} 秒")
print(f"总帧数: {total_frames}")
print(f"类型: {type(fps)}")

# 字符串 (str)
filename = "test_image.jpg"
folder = "E:/images"
full_path = folder + "/" + filename
print(f"\n文件名: {filename}")
print(f"完整路径: {full_path}")
print(f"类型: {type(filename)}")

# 字符串常用操作
text = "  Hello OpenCV  "
print(f"\n字符串操作:")
print(f"  原始: '{text}'")
print(f"  去空格: '{text.strip()}'")
print(f"  大写: '{text.upper()}'")
print(f"  小写: '{text.lower()}'")
print(f"  替换: '{text.replace('OpenCV', 'Python')}'")
print(f"  分割: {full_path.split('/')}")

# 布尔值 (bool)
is_color = True
has_alpha = False
print(f"\n是否彩色: {is_color}")
print(f"有透明通道: {has_alpha}")
print(f"类型: {type(is_color)}")

# 空值 (NoneType)
result = None
print(f"\n结果: {result}")
print(f"类型: {type(result)}")

# ========== 2. 列表 (list) ==========
print("\n" + "=" * 50)
print("【2. 列表操作】")

# 创建列表
pixels = [255, 128, 64, 32, 0]
print(f"像素值列表: {pixels}")

# 访问元素
print(f"第一个元素: {pixels[0]}")
print(f"最后一个元素: {pixels[-1]}")
print(f"前三个元素: {pixels[:3]}")
print(f"后两个元素: {pixels[-2:]}")

# 修改元素
pixels[0] = 200
print(f"修改后: {pixels}")

# 添加元素
pixels.append(16)
print(f"添加后: {pixels}")

# 插入元素
pixels.insert(0, 250)
print(f"插入后: {pixels}")

# 删除元素
removed = pixels.pop()
print(f"弹出元素: {removed}")
print(f"弹出后: {pixels}")

# 列表长度
print(f"列表长度: {len(pixels)}")

# 列表排序
pixels.sort()
print(f"升序排序: {pixels}")
pixels.sort(reverse=True)
print(f"降序排序: {pixels}")

# 列表推导式
squares = [x**2 for x in range(5)]
print(f"平方列表: {squares}")

# 条件列表推导式
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"偶数平方: {even_squares}")

# ========== 3. 元组 (tuple) ==========
print("\n" + "=" * 50)
print("【3. 元组操作】")

# 创建元组（不可变）
image_size = (640, 480)
color_rgb = (255, 128, 0)
print(f"图像尺寸: {image_size}")
print(f"RGB颜色: {color_rgb}")

# 元组解包
w, h = image_size
r, g, b = color_rgb
print(f"宽度: {w}, 高度: {h}")
print(f"R: {r}, G: {g}, B: {b}")

# 元组作为字典键（列表不能）
point_colors = {
    (0, 0): "red",
    (100, 100): "blue",
    (200, 200): "green"
}
print(f"点(0,0)的颜色: {point_colors[(0, 0)]}")

# ========== 4. 字典 (dict) ==========
print("\n" + "=" * 50)
print("【4. 字典操作】")

# 创建字典
image_info = {
    "width": 640,
    "height": 480,
    "channels": 3,
    "format": "RGB",
    "depth": 8
}
print(f"图像信息: {image_info}")

# 访问值
print(f"宽度: {image_info['width']}")
print(f"格式: {image_info.get('format', 'unknown')}")

# 添加/修改
image_info["compression"] = "JPEG"
image_info["width"] = 800
print(f"更新后: {image_info}")

# 删除
del image_info["compression"]
print(f"删除后: {image_info}")

# 遍历字典
print("\n遍历字典:")
for key, value in image_info.items():
    print(f"  {key}: {value}")

# 获取所有键和值
print(f"\n所有键: {list(image_info.keys())}")
print(f"所有值: {list(image_info.values())}")

# 字典推导式
squared_dict = {x: x**2 for x in range(5)}
print(f"字典推导式: {squared_dict}")

# ========== 5. 集合 (set) ==========
print("\n" + "=" * 50)
print("【5. 集合操作】")

# 创建集合（无序、不重复）
colors1 = {"red", "green", "blue"}
colors2 = {"green", "yellow", "blue"}
print(f"集合1: {colors1}")
print(f"集合2: {colors2}")

# 集合运算
print(f"并集: {colors1 | colors2}")
print(f"交集: {colors1 & colors2}")
print(f"差集: {colors1 - colors2}")
print(f"对称差集: {colors1 ^ colors2}")

# 去重
numbers_with_dup = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique_numbers = list(set(numbers_with_dup))
print(f"去重前: {numbers_with_dup}")
print(f"去重后: {unique_numbers}")

# ========== 6. 类型转换 ==========
print("\n" + "=" * 50)
print("【6. 类型转换】")

# 字符串转数字
str_num = "42"
int_num = int(str_num)
float_num = float(str_num)
print(f"字符串 '{str_num}' -> 整数 {int_num} -> 浮点 {float_num}")

# 数字转字符串
num = 3.14159
str_val = str(num)
print(f"数字 {num} -> 字符串 '{str_val}'")

# 列表与元组互转
lst = [1, 2, 3]
tup = tuple(lst)
back_to_list = list(tup)
print(f"列表 {lst} -> 元组 {tup} -> 列表 {back_to_list}")

# 布尔转换
print(f"\nbool(0) = {bool(0)}")
print(f"bool(1) = {bool(1)}")
print(f"bool('') = {bool('')}")
print(f"bool('hello') = {bool('hello')}")
print(f"bool([]) = {bool([])}")
print(f"bool([1,2]) = {bool([1,2])}")

# ========== 7. 类型检查 ==========
print("\n" + "=" * 50)
print("【7. 类型检查】")

value = 42
print(f"type({value}) = {type(value)}")
print(f"isinstance({value}, int) = {isinstance(value, int)}")
print(f"isinstance({value}, (int, float)) = {isinstance(value, (int, float))}")

print("\n" + "=" * 50)
print("示例代码运行完成！")
