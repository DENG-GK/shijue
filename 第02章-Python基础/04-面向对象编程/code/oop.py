# -*- coding: utf-8 -*-
"""
面向对象编程 - 示例代码
本脚本演示Python类与对象的核心概念
"""

print("=" * 50)
print("Python 面向对象编程示例")
print("=" * 50)

# ========== 1. 基本类定义 ==========
print("\n【1. 基本类定义】")

class Image:
    """图像类 - 模拟一个简单的图像对象"""

    def __init__(self, width, height, channels=3):
        """构造函数"""
        self.width = width
        self.height = height
        self.channels = channels
        # 模拟像素数据（用列表代替numpy数组）
        self.data = [0] * (width * height * channels)

    def get_size(self):
        """获取图像尺寸"""
        return (self.width, self.height)

    def get_pixel_count(self):
        """获取像素总数"""
        return self.width * self.height

    def __str__(self):
        """字符串表示"""
        return f"Image({self.width}x{self.height}, {self.channels}ch)"

    def __repr__(self):
        """官方字符串表示"""
        return f"Image(width={self.width}, height={self.height}, channels={self.channels})"

# 创建对象
img = Image(640, 480)
print(f"图像: {img}")
print(f"尺寸: {img.get_size()}")
print(f"像素数: {img.get_pixel_count()}")

# 灰度图
gray_img = Image(320, 240, channels=1)
print(f"灰度图: {gray_img}")

# ========== 2. 属性访问控制 ==========
print("\n【2. 属性访问控制】")

class Camera:
    """相机类 - 演示属性访问控制"""

    def __init__(self, name, resolution):
        self.name = name                  # 公有属性
        self._resolution = resolution     # 受保护属性（约定）
        self.__serial = "SN-12345"        # 私有属性（名称改编）

    @property
    def resolution(self):
        """使用property装饰器获取分辨率"""
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        """设置分辨率（带验证）"""
        w, h = value
        if w > 0 and h > 0:
            self._resolution = value
        else:
            raise ValueError("分辨率必须为正数")

    def get_info(self):
        return f"{self.name} - {self._resolution[0]}x{self._resolution[1]}"

cam = Camera("WebCam", (1920, 1080))
print(f"相机: {cam.get_info()}")
print(f"分辨率: {cam.resolution}")

# 使用setter
cam.resolution = (3840, 2160)
print(f"更新后: {cam.get_info()}")

# ========== 3. 继承 ==========
print("\n【3. 继承】")

class Shape:
    """基类 - 图形"""

    def __init__(self, name, color="black"):
        self.name = name
        self.color = color

    def area(self):
        """计算面积（子类需要重写）"""
        return 0

    def perimeter(self):
        """计算周长（子类需要重写）"""
        return 0

    def __str__(self):
        return f"{self.name}(颜色={self.color})"

class Rectangle(Shape):
    """矩形 - 继承自Shape"""

    def __init__(self, width, height, color="black"):
        super().__init__("矩形", color)  # 调用父类构造函数
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

    def __str__(self):
        return f"矩形({self.width}x{self.height}, 颜色={self.color})"

class Circle(Shape):
    """圆形 - 继承自Shape"""

    def __init__(self, radius, color="black"):
        super().__init__("圆形", color)
        self.radius = radius

    def area(self):
        import math
        return math.pi * self.radius ** 2

    def perimeter(self):
        import math
        return 2 * math.pi * self.radius

    def __str__(self):
        return f"圆形(半径={self.radius}, 颜色={self.color})"

# 使用
rect = Rectangle(100, 50, "red")
circle = Circle(30, "blue")
print(f"{rect} -> 面积={rect.area()}, 周长={rect.perimeter()}")
print(f"{circle} -> 面积={circle.area():.2f}, 周长={circle.perimeter():.2f}")

# ========== 4. 多态 ==========
print("\n【4. 多态】")

def print_shape_info(shape):
    """多态 - 同一函数处理不同类型的Shape"""
    print(f"  {shape} -> 面积: {shape.area():.2f}")

shapes = [
    Rectangle(10, 20, "red"),
    Circle(15, "blue"),
    Rectangle(30, 40, "green"),
    Circle(5, "yellow")
]

print("所有图形信息:")
for s in shapes:
    print_shape_info(s)

# 按面积排序
shapes.sort(key=lambda s: s.area())
print("\n按面积排序:")
for s in shapes:
    print_shape_info(s)

# ========== 5. 魔术方法 ==========
print("\n【5. 魔术方法】")

class Pixel:
    """像素类 - 演示魔术方法"""

    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __str__(self):
        return f"Pixel(R={self.r}, G={self.g}, B={self.b})"

    def __repr__(self):
        return f"Pixel({self.r}, {self.g}, {self.b})"

    def __eq__(self, other):
        """相等比较"""
        if isinstance(other, Pixel):
            return self.r == other.r and self.g == other.g and self.b == other.b
        return False

    def __add__(self, other):
        """像素加法（饱和加法）"""
        return Pixel(
            min(255, self.r + other.r),
            min(255, self.g + other.g),
            min(255, self.b + other.b)
        )

    def __mul__(self, scalar):
        """像素乘法（标量）"""
        return Pixel(
            min(255, int(self.r * scalar)),
            min(255, int(self.g * scalar)),
            min(255, int(self.b * scalar))
        )

    def __len__(self):
        """通道数"""
        return 3

    def __getitem__(self, index):
        """索引访问"""
        channels = [self.r, self.g, self.b]
        return channels[index]

    def brightness(self):
        """计算亮度"""
        return (self.r + self.g + self.b) / 3

p1 = Pixel(100, 150, 200)
p2 = Pixel(50, 80, 100)
print(f"p1 = {p1}")
print(f"p2 = {p2}")
print(f"p1 + p2 = {p1 + p2}")
print(f"p1 * 0.5 = {p1 * 0.5}")
print(f"p1 == p2: {p1 == p2}")
print(f"len(p1) = {len(p1)}")
print(f"p1[0] = {p1[0]} (R通道)")
print(f"p1亮度 = {p1.brightness():.1f}")

# ========== 6. 类方法和静态方法 ==========
print("\n【6. 类方法和静态方法】")

class Color:
    """颜色类 - 演示类方法和静态方法"""

    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    @classmethod
    def from_hex(cls, hex_string):
        """类方法：从十六进制字符串创建颜色"""
        hex_string = hex_string.lstrip('#')
        r = int(hex_string[0:2], 16)
        g = int(hex_string[2:4], 16)
        b = int(hex_string[4:6], 16)
        return cls(r, g, b)

    @classmethod
    def red(cls):
        """类方法：预定义红色"""
        return cls(255, 0, 0)

    @classmethod
    def green(cls):
        return cls(0, 255, 0)

    @staticmethod
    def is_valid_value(value):
        """静态方法：检查是否是有效的颜色值"""
        return 0 <= value <= 255

    def __str__(self):
        return f"Color({self.r}, {self.g}, {self.b})"

# 类方法
c1 = Color.from_hex("#FF8000")
c2 = Color.red()
print(f"从十六进制: {c1}")
print(f"预定义红色: {c2}")

# 静态方法
print(f"128是有效值: {Color.is_valid_value(128)}")
print(f"300是有效值: {Color.is_valid_value(300)}")

# ========== 7. 实际应用示例 ==========
print("\n【7. 实际应用 - 图像处理器】")

class ImageProcessor:
    """图像处理器类"""

    def __init__(self, name="默认处理器"):
        self.name = name
        self._operations = []

    def add_operation(self, op_name, params=None):
        """添加处理操作"""
        self._operations.append({"name": op_name, "params": params or {}})
        return self  # 链式调用

    def process(self, image_data):
        """执行所有操作"""
        result = image_data.copy()
        for op in self._operations:
            print(f"  执行: {op['name']} {op['params']}")
        return result

    def clear(self):
        """清除操作列表"""
        self._operations.clear()

    def __len__(self):
        return len(self._operations)

    def __str__(self):
        return f"ImageProcessor('{self.name}', {len(self._operations)}个操作)"

# 链式调用
processor = ImageProcessor("视觉处理器")
processor.add_operation("灰度化").add_operation("高斯模糊", {"ksize": 5}).add_operation("边缘检测", {"method": "Canny"})

print(f"处理器: {processor}")
print(f"操作数: {len(processor)}")

# 模拟处理
print("开始处理:")
processor.process([128, 64, 32])

print("\n" + "=" * 50)
print("示例代码运行完成！")
