# -*- coding: utf-8 -*-
"""
文件操作 - 示例代码
本脚本演示Python文件读写和路径操作
"""

import os
import csv
import json
from pathlib import Path

print("=" * 50)
print("Python 文件操作示例")
print("=" * 50)

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 1. 文本文件写入 ==========
print("\n【1. 文本文件写入】")

# 写入文件
test_file = os.path.join(SCRIPT_DIR, "test_output.txt")
with open(test_file, "w", encoding="utf-8") as f:
    f.write("第一行：计算机视觉学习笔记\n")
    f.write("第二行：Python文件操作\n")
    f.write("第三行：OpenCV图像处理\n")
print(f"已写入文件: {test_file}")

# 追加写入
with open(test_file, "a", encoding="utf-8") as f:
    f.write("第四行：追加的内容\n")
print("已追加内容")

# ========== 2. 文本文件读取 ==========
print("\n【2. 文本文件读取】")

# 方法1：read() 一次性读取全部
with open(test_file, "r", encoding="utf-8") as f:
    content = f.read()
print("read() 读取全部:")
print(content)

# 方法2：readline() 逐行读取
print("readline() 逐行读取:")
with open(test_file, "r", encoding="utf-8") as f:
    line = f.readline()
    while line:
        print(f"  {line.strip()}")
        line = f.readline()

# 方法3：readlines() 读取所有行到列表
with open(test_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
print(f"\nreadlines() 总行数: {len(lines)}")

# 方法4：直接遍历文件对象（推荐）
print("\n直接遍历文件对象:")
with open(test_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(f"  行{i}: {line.strip()}")

# ========== 3. CSV 文件处理 ==========
print("\n【3. CSV 文件处理】")

# 写入 CSV
csv_file = os.path.join(SCRIPT_DIR, "image_data.csv")
data = [
    ["文件名", "宽度", "高度", "通道数"],
    ["img001.jpg", 640, 480, 3],
    ["img002.png", 800, 600, 4],
    ["img003.jpg", 1920, 1080, 3],
    ["img004.bmp", 320, 240, 1],
]

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(data)
print(f"已写入CSV: {csv_file}")

# 读取 CSV
print("\n读取CSV内容:")
with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  # 读取表头
    print(f"  表头: {header}")
    for row in reader:
        print(f"  {row[0]}: {row[1]}x{row[2]}, {row[3]}通道")

# 使用 DictReader
print("\n使用DictReader:")
with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  {row['文件名']} -> {row['宽度']}x{row['高度']}")

# ========== 4. JSON 文件处理 ==========
print("\n【4. JSON 文件处理】")

# 写入 JSON
json_file = os.path.join(SCRIPT_DIR, "config.json")
config = {
    "project": "计算机视觉教程",
    "version": "1.0",
    "image_settings": {
        "width": 640,
        "height": 480,
        "channels": 3,
        "format": "RGB"
    },
    "preprocessing": [
        {"name": "resize", "params": {"scale": 0.5}},
        {"name": "normalize", "params": {"mean": [0.485, 0.456, 0.406]}},
        {"name": "grayscale", "params": {}}
    ],
    "gpu_enabled": False
}

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"已写入JSON: {json_file}")

# 读取 JSON
with open(json_file, "r", encoding="utf-8") as f:
    loaded_config = json.load(f)

print(f"项目: {loaded_config['project']}")
print(f"图像设置: {loaded_config['image_settings']}")
print(f"预处理步骤:")
for step in loaded_config['preprocessing']:
    print(f"  - {step['name']}: {step['params']}")

# JSON 字符串转换
json_str = json.dumps(config, ensure_ascii=False)
print(f"\nJSON字符串长度: {len(json_str)}")

# ========== 5. 路径操作 - os.path ==========
print("\n【5. 路径操作 - os.path】")

filepath = os.path.join("E:", "images", "test", "photo.jpg")
print(f"拼接路径: {filepath}")
print(f"目录名: {os.path.dirname(filepath)}")
print(f"文件名: {os.path.basename(filepath)}")
print(f"文件名（无扩展名）: {os.path.splitext(os.path.basename(filepath))[0]}")
print(f"扩展名: {os.path.splitext(filepath)[1]}")
print(f"是否绝对路径: {os.path.isabs(filepath)}")

# 当前目录信息
print(f"\n当前工作目录: {os.getcwd()}")
print(f"脚本目录: {SCRIPT_DIR}")

# ========== 6. 路径操作 - pathlib（推荐） ==========
print("\n【6. 路径操作 - pathlib】")

p = Path("E:/images/test/photo.jpg")
print(f"路径: {p}")
print(f"父目录: {p.parent}")
print(f"文件名: {p.name}")
print(f"文件名（无后缀）: {p.stem}")
print(f"后缀: {p.suffix}")
print(f"各部分: {p.parts}")

# 路径拼接
new_path = Path("E:/images") / "output" / "result.png"
print(f"拼接: {new_path}")

# 修改后缀
png_path = p.with_suffix('.png')
print(f"改后缀: {png_path}")

# 修改文件名
renamed = p.with_name("new_photo.jpg")
print(f"改文件名: {renamed}")

# 遍历目录中的文件
print(f"\n脚本目录下的文件:")
for f in Path(SCRIPT_DIR).iterdir():
    file_type = "目录" if f.is_dir() else "文件"
    print(f"  [{file_type}] {f.name}")

# ========== 7. 异常处理 ==========
print("\n【7. 异常处理】")

# try-except-else-finally
def safe_read_file(filepath):
    """安全读取文件"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  文件不存在: {filepath}")
        return None
    except PermissionError:
        print(f"  权限不足: {filepath}")
        return None
    except Exception as e:
        print(f"  未知错误: {e}")
        return None
    else:
        print(f"  成功读取 {len(content)} 个字符")
        return content
    finally:
        print(f"  文件操作完成")

# 测试
print("读取存在的文件:")
safe_read_file(test_file)

print("\n读取不存在的文件:")
safe_read_file("nonexistent_file.txt")

# 自定义异常
class ImageLoadError(Exception):
    """图像加载错误"""
    def __init__(self, filepath, reason="未知原因"):
        self.filepath = filepath
        self.reason = reason
        super().__init__(f"无法加载图像 '{filepath}': {reason}")

print("\n自定义异常:")
try:
    raise ImageLoadError("broken.jpg", "文件格式损坏")
except ImageLoadError as e:
    print(f"  捕获异常: {e}")

# ========== 清理测试文件 ==========
print("\n【清理测试文件】")
for f in [test_file, csv_file, json_file]:
    if os.path.exists(f):
        os.remove(f)
        print(f"  已删除: {os.path.basename(f)}")

print("\n" + "=" * 50)
print("示例代码运行完成！")
