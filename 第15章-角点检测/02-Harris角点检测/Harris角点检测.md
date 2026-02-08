# Harris角点检测

## 1. Harris角点检测概述

Harris角点检测是由Chris Harris和Mike Stephens于1988年提出的经典角点检测算法。它基于自相关矩阵的特征值分析，是计算机视觉中最广泛使用的角点检测方法之一。

### 1.1 算法原理

Harris角点检测的核心思想：
1. 计算图像梯度 $I_x$ 和 $I_y$
2. 构建自相关矩阵 $M$
3. 计算Harris响应函数 $R$
4. 非极大值抑制

**Harris响应函数**：

$$R = \det(M) - k \cdot \text{trace}(M)^2$$

其中：
- $\det(M) = \lambda_1 \cdot \lambda_2$
- $\text{trace}(M) = \lambda_1 + \lambda_2$
- $k$ 是经验常数，通常取0.04-0.06

### 1.2 响应值分析

| R值 | 含义 |
|-----|------|
| R >> 0 | 角点 |
| R << 0 | 边缘 |
| |R| ≈ 0 | 平坦区域 |

### 1.3 cv2.cornerHarris() 函数

```python
dst = cv2.cornerHarris(src, blockSize, ksize, k)
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| src | 输入图像（浮点型） |
| blockSize | 邻域大小 |
| ksize | Sobel算子孔径 |
| k | Harris参数 |
| dst | Harris响应图 |

## 2. 代码示例

### 示例1：基本Harris角点检测

```python
"""
示例1：基本Harris角点检测
使用cv2.cornerHarris()检测角点
"""
import cv2
import numpy as np

# 创建测试图像
img = np.zeros((400, 500), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), 200, -1)
cv2.rectangle(img, (250, 80), (450, 300), 200, -1)
cv2.circle(img, (150, 320), 60, 200, -1)

# 转换为浮点数
img_float = img.astype(np.float32)

# Harris角点检测
dst = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)

# 膨胀角点响应
dst = cv2.dilate(dst, None)

# 阈值检测
threshold = 0.01 * dst.max()

# 创建彩色结果图像
result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 标记角点
result[dst > threshold] = [0, 0, 255]

# 统计角点数量
corner_count = np.sum(dst > threshold)
print(f"检测到 {corner_count} 个角点像素")

cv2.imshow("Original", img)
cv2.imshow("Harris Response", cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
cv2.imshow("Corners Detected", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例2：Harris角点检测完整流程

```python
"""
示例2：Harris角点检测完整流程
展示从图像读取到角点标记的完整过程
"""
import cv2
import numpy as np

def harris_corner_detection(img, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """
    完整的Harris角点检测流程

    Args:
        img: 输入图像（灰度或彩色）
        block_size: 邻域大小
        ksize: Sobel算子孔径
        k: Harris参数
        threshold: 阈值（相对于最大响应）

    Returns:
        corners: 角点坐标列表 [(x, y), ...]
        response: Harris响应图
    """
    # 转灰度
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 转浮点
    gray_float = gray.astype(np.float32)

    # Harris角点检测
    response = cv2.cornerHarris(gray_float, block_size, ksize, k)

    # 非极大值抑制
    # 使用膨胀找局部最大值
    dilated = cv2.dilate(response, None)
    local_max = (response == dilated)

    # 阈值
    thresh_value = threshold * response.max()
    corners_mask = (response > thresh_value) & local_max

    # 获取角点坐标
    corners = np.where(corners_mask)
    corner_list = list(zip(corners[1], corners[0]))  # (x, y)

    return corner_list, response

# 测试图像
img = cv2.imread("image.jpg")
if img is None:
    # 创建测试图像
    img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    cv2.rectangle(img, (50, 50), (200, 200), (50, 50, 50), -1)
    cv2.rectangle(img, (300, 100), (550, 300), (80, 80, 80), -1)
    cv2.fillPoly(img, [np.array([[100, 280], [200, 350], [50, 380]])], (100, 100, 100))

# 检测角点
corners, response = harris_corner_detection(img, threshold=0.01)
print(f"检测到 {len(corners)} 个角点")

# 可视化
result = img.copy()
for x, y in corners:
    cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

# 显示响应图
response_vis = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
response_color = cv2.applyColorMap(response_vis, cv2.COLORMAP_JET)

cv2.imshow("Original", img)
cv2.imshow("Harris Response", response_color)
cv2.imshow("Detected Corners", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例3：参数调节对比

```python
"""
示例3：参数调节对比
比较不同参数设置的检测效果
"""
import cv2
import numpy as np

# 创建测试图像
img = np.zeros((250, 300), dtype=np.uint8)
cv2.rectangle(img, (30, 30), (120, 120), 200, -1)
cv2.rectangle(img, (150, 50), (270, 150), 200, -1)
cv2.circle(img, (80, 190), 40, 200, -1)

img_float = img.astype(np.float32)

# 参数组合
params = [
    {"blockSize": 2, "ksize": 3, "k": 0.04, "name": "Default"},
    {"blockSize": 2, "ksize": 3, "k": 0.02, "name": "k=0.02"},
    {"blockSize": 2, "ksize": 3, "k": 0.10, "name": "k=0.10"},
    {"blockSize": 3, "ksize": 3, "k": 0.04, "name": "block=3"},
    {"blockSize": 5, "ksize": 3, "k": 0.04, "name": "block=5"},
    {"blockSize": 2, "ksize": 5, "k": 0.04, "name": "ksize=5"},
]

print("不同参数设置的检测结果:")
print("-" * 50)
print(f"{'参数':>15} {'角点数':>10} {'响应范围':>20}")
print("-" * 50)

results = []
for p in params:
    # 检测
    dst = cv2.cornerHarris(img_float, p["blockSize"], p["ksize"], p["k"])

    # 阈值
    threshold = 0.01 * dst.max() if dst.max() > 0 else 0
    corners = dst > threshold

    # 膨胀显示
    corners_dilated = cv2.dilate(corners.astype(np.uint8), None)

    corner_count = np.sum(corners)
    print(f"{p['name']:>15} {corner_count:>10} [{dst.min():.2e}, {dst.max():.2e}]")

    # 可视化
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    canvas[corners_dilated > 0] = [0, 0, 255]
    cv2.putText(canvas, p["name"], (5, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    results.append(canvas)

# 组合显示
row1 = np.hstack(results[:3])
row2 = np.hstack(results[3:])
combined = np.vstack([row1, row2])

cv2.imshow("Parameter Comparison", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例4：非极大值抑制

```python
"""
示例4：非极大值抑制
实现更精确的角点定位
"""
import cv2
import numpy as np

def non_maximum_suppression(response, window_size=5, threshold=0.01):
    """
    非极大值抑制

    Args:
        response: Harris响应图
        window_size: 抑制窗口大小
        threshold: 响应阈值（相对于最大值）

    Returns:
        corners: 角点坐标列表
    """
    # 计算阈值
    thresh_value = threshold * response.max()

    # 候选角点
    candidates = response > thresh_value

    # 创建结果掩码
    corners = np.zeros_like(response, dtype=bool)

    # 获取候选点坐标
    y_coords, x_coords = np.where(candidates)

    # 对每个候选点进行非极大值抑制
    half = window_size // 2
    h, w = response.shape

    for y, x in zip(y_coords, x_coords):
        # 定义邻域范围
        y1 = max(0, y - half)
        y2 = min(h, y + half + 1)
        x1 = max(0, x - half)
        x2 = min(w, x + half + 1)

        # 获取邻域
        neighborhood = response[y1:y2, x1:x2]

        # 检查是否是局部最大值
        if response[y, x] == neighborhood.max():
            corners[y, x] = True

    # 返回角点坐标
    y_corners, x_corners = np.where(corners)
    return list(zip(x_corners, y_corners))

# 创建测试图像
img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
cv2.rectangle(img, (200, 80), (350, 220), 200, -1)

# 计算Harris响应
img_float = img.astype(np.float32)
response = cv2.cornerHarris(img_float, 2, 3, 0.04)

# 不同窗口大小的NMS
window_sizes = [3, 5, 7, 11]

print("非极大值抑制效果:")
print("-" * 40)
print(f"{'窗口大小':>10} {'角点数':>10}")
print("-" * 40)

results = []
for ws in window_sizes:
    corners = non_maximum_suppression(response, ws, 0.01)
    print(f"{ws:>10} {len(corners):>10}")

    # 可视化
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in corners:
        cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

    cv2.putText(canvas, f"NMS window={ws}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(canvas, f"Corners: {len(corners)}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    results.append(canvas)

# 组合显示
row1 = np.hstack(results[:2])
row2 = np.hstack(results[2:])
combined = np.vstack([row1, row2])

cv2.imshow("NMS Comparison", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例5：真实图像角点检测

```python
"""
示例5：真实图像角点检测
在真实图像上进行角点检测
"""
import cv2
import numpy as np

# 读取图像
img = cv2.imread("building.jpg")
if img is None:
    # 创建模拟建筑图像
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200

    # 建筑轮廓
    cv2.rectangle(img, (100, 100), (300, 350), (150, 150, 150), -1)
    cv2.rectangle(img, (350, 150), (500, 350), (130, 130, 130), -1)

    # 窗户
    for i in range(3):
        for j in range(4):
            cv2.rectangle(img, (120 + i*60, 120 + j*55),
                         (160 + i*60, 160 + j*55), (80, 80, 80), -1)

    for i in range(2):
        for j in range(3):
            cv2.rectangle(img, (370 + i*60, 170 + j*55),
                         (410 + i*60, 210 + j*55), (80, 80, 80), -1)

    # 门
    cv2.rectangle(img, (180, 280), (220, 350), (60, 60, 60), -1)

    # 添加一些纹理
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# 转灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris角点检测
gray_float = gray.astype(np.float32)
dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)

# 膨胀
dst = cv2.dilate(dst, None)

# 不同阈值
thresholds = [0.001, 0.01, 0.05, 0.1]

print("不同阈值的检测结果:")
print("-" * 40)

results = []
for thresh in thresholds:
    corners = dst > thresh * dst.max()
    corner_count = np.sum(corners)

    print(f"阈值 {thresh}: {corner_count} 个角点")

    result = img.copy()
    result[corners] = [0, 0, 255]

    cv2.putText(result, f"thresh={thresh}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(result, f"corners={corner_count}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    results.append(result)

# 组合显示
row1 = np.hstack(results[:2])
row2 = np.hstack(results[2:])
combined = np.vstack([row1, row2])

cv2.imshow("Threshold Comparison", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例6：Harris角点的旋转不变性

```python
"""
示例6：Harris角点的旋转不变性
验证Harris角点检测对旋转的不变性
"""
import cv2
import numpy as np

def detect_harris_corners(img, threshold=0.01):
    """检测Harris角点并返回坐标列表"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray_float = gray.astype(np.float32)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)

    # NMS
    dilated = cv2.dilate(dst, None)
    local_max = (dst == dilated)
    corners_mask = (dst > threshold * dst.max()) & local_max

    y, x = np.where(corners_mask)
    return list(zip(x, y))

def rotate_image(img, angle):
    """旋转图像"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=200)
    return rotated, M

# 创建测试图像
img = np.ones((300, 300), dtype=np.uint8) * 200
cv2.rectangle(img, (80, 80), (220, 220), 100, -1)

# 原始角点
original_corners = detect_harris_corners(img)
print(f"原始图像检测到 {len(original_corners)} 个角点")

# 不同旋转角度
angles = [0, 30, 45, 60, 90, 180]

results = []
corner_counts = []

for angle in angles:
    if angle == 0:
        rotated = img.copy()
        corners = original_corners
    else:
        rotated, M = rotate_image(img, angle)
        corners = detect_harris_corners(rotated)

    corner_counts.append(len(corners))

    # 可视化
    canvas = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    for x, y in corners:
        cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

    cv2.putText(canvas, f"Angle: {angle}deg", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(canvas, f"Corners: {len(corners)}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    results.append(canvas)

# 统计
print("\n旋转不变性测试:")
print("-" * 40)
for angle, count in zip(angles, corner_counts):
    print(f"旋转 {angle:>3}°: {count} 个角点")

# 组合显示
row1 = np.hstack(results[:3])
row2 = np.hstack(results[3:])
combined = np.vstack([row1, row2])

cv2.imshow("Rotation Invariance", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 结论
if max(corner_counts) - min(corner_counts) <= 2:
    print("\n结论: Harris角点检测具有良好的旋转不变性")
else:
    print("\n注意: 边界效应可能影响角点数量")
```

### 示例7：Harris角点与边缘的区分

```python
"""
示例7：Harris角点与边缘的区分
可视化Harris响应值的正负区域
"""
import cv2
import numpy as np

# 创建包含边缘和角点的图像
img = np.zeros((300, 400), dtype=np.uint8)

# 矩形（有角点和边缘）
cv2.rectangle(img, (50, 50), (150, 150), 200, -1)

# 圆形（只有边缘，无角点）
cv2.circle(img, (280, 100), 50, 200, -1)

# 线条（边缘）
cv2.line(img, (50, 220), (350, 280), 200, 5)

# 三角形（有角点）
pts = np.array([[200, 200], [280, 280], [150, 280]])
cv2.fillPoly(img, [pts], 200)

# Harris响应
img_float = img.astype(np.float32)
response = cv2.cornerHarris(img_float, 2, 3, 0.04)

# 分离正负响应
positive_response = np.maximum(response, 0)
negative_response = np.minimum(response, 0)

# 归一化
pos_norm = cv2.normalize(positive_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
neg_norm = cv2.normalize(np.abs(negative_response), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 创建彩色可视化
# 角点（正响应）= 红色
# 边缘（负响应）= 蓝色
visualization = np.zeros((300, 400, 3), dtype=np.uint8)
visualization[:, :, 2] = pos_norm  # 红色通道 = 角点
visualization[:, :, 0] = neg_norm  # 蓝色通道 = 边缘

# 阈值检测
corner_threshold = 0.01 * response.max()
edge_threshold = 0.01 * response.min()

corners = response > corner_threshold
edges = response < edge_threshold

# 统计
print("Harris响应分析:")
print("-" * 40)
print(f"响应范围: [{response.min():.2e}, {response.max():.2e}]")
print(f"角点像素 (R > {corner_threshold:.2e}): {np.sum(corners)}")
print(f"边缘像素 (R < {edge_threshold:.2e}): {np.sum(edges)}")

# 创建标注结果
result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
result[corners] = [0, 0, 255]  # 角点红色
result[edges] = [255, 0, 0]    # 边缘蓝色

# 显示
cv2.imshow("Original", img)
cv2.imshow("Harris Response Visualization", visualization)
cv2.imshow("Corners (Red) and Edges (Blue)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例8：Harris角点检测器类

```python
"""
示例8：Harris角点检测器类
封装Harris角点检测的完整功能
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional

class HarrisCornerDetector:
    """Harris角点检测器"""

    def __init__(self,
                 block_size: int = 2,
                 ksize: int = 3,
                 k: float = 0.04,
                 threshold: float = 0.01,
                 nms_window: int = 5):
        """
        初始化检测器

        Args:
            block_size: 邻域大小
            ksize: Sobel算子孔径
            k: Harris参数
            threshold: 响应阈值（相对于最大值）
            nms_window: 非极大值抑制窗口
        """
        self.block_size = block_size
        self.ksize = ksize
        self.k = k
        self.threshold = threshold
        self.nms_window = nms_window

        self.response = None
        self.corners = None

    def detect(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """
        检测角点

        Args:
            img: 输入图像

        Returns:
            角点坐标列表 [(x, y), ...]
        """
        # 转灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 转浮点
        gray_float = gray.astype(np.float32)

        # Harris响应
        self.response = cv2.cornerHarris(gray_float, self.block_size,
                                          self.ksize, self.k)

        # 非极大值抑制
        dilated = cv2.dilate(self.response, None, iterations=1)
        local_max = (self.response == dilated)

        # 阈值
        thresh_value = self.threshold * self.response.max()
        corners_mask = (self.response > thresh_value) & local_max

        # NMS with window
        if self.nms_window > 1:
            corners_mask = self._nms_with_window(corners_mask)

        # 获取坐标
        y, x = np.where(corners_mask)
        self.corners = list(zip(x, y))

        return self.corners

    def _nms_with_window(self, corners_mask: np.ndarray) -> np.ndarray:
        """窗口化非极大值抑制"""
        result = np.zeros_like(corners_mask)
        y_coords, x_coords = np.where(corners_mask)

        half = self.nms_window // 2
        h, w = corners_mask.shape

        for y, x in zip(y_coords, x_coords):
            y1, y2 = max(0, y-half), min(h, y+half+1)
            x1, x2 = max(0, x-half), min(w, x+half+1)

            neighborhood = self.response[y1:y2, x1:x2]
            if self.response[y, x] == neighborhood.max():
                result[y, x] = True

        return result

    def get_response_map(self) -> Optional[np.ndarray]:
        """获取Harris响应图"""
        return self.response

    def visualize(self,
                  img: np.ndarray,
                  radius: int = 5,
                  color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        可视化检测结果

        Args:
            img: 原始图像
            radius: 角点标记半径
            color: 角点颜色

        Returns:
            标记了角点的图像
        """
        if self.corners is None:
            self.detect(img)

        result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for x, y in self.corners:
            cv2.circle(result, (x, y), radius, color, -1)

        return result

    def get_corner_strengths(self) -> List[Tuple[int, int, float]]:
        """
        获取角点及其响应强度

        Returns:
            [(x, y, strength), ...]
        """
        if self.corners is None or self.response is None:
            return []

        result = []
        for x, y in self.corners:
            strength = self.response[y, x]
            result.append((x, y, strength))

        # 按强度排序
        result.sort(key=lambda x: x[2], reverse=True)
        return result

    def filter_by_strength(self, min_strength: float = None,
                           max_corners: int = None) -> List[Tuple[int, int]]:
        """
        按强度筛选角点

        Args:
            min_strength: 最小强度（绝对值）
            max_corners: 最大角点数

        Returns:
            筛选后的角点列表
        """
        corners_with_strength = self.get_corner_strengths()

        if min_strength is not None:
            corners_with_strength = [(x, y, s) for x, y, s in corners_with_strength
                                     if s >= min_strength]

        if max_corners is not None:
            corners_with_strength = corners_with_strength[:max_corners]

        return [(x, y) for x, y, _ in corners_with_strength]


# 使用示例
if __name__ == "__main__":
    # 创建测试图像
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)
    cv2.rectangle(img, (50, 50), (200, 200), (100, 100, 100), -1)
    cv2.rectangle(img, (250, 80), (450, 300), (80, 80, 80), -1)
    cv2.circle(img, (150, 320), 60, (120, 120, 120), -1)

    # 创建检测器
    detector = HarrisCornerDetector(threshold=0.01, nms_window=5)

    # 检测
    corners = detector.detect(img)
    print(f"检测到 {len(corners)} 个角点")

    # 获取带强度的角点
    corners_with_strength = detector.get_corner_strengths()
    print("\n最强的5个角点:")
    for x, y, s in corners_with_strength[:5]:
        print(f"  ({x}, {y}): {s:.4f}")

    # 筛选
    top_corners = detector.filter_by_strength(max_corners=10)
    print(f"\n筛选后保留 {len(top_corners)} 个角点")

    # 可视化
    result = detector.visualize(img)

    # 显示响应图
    response = detector.get_response_map()
    response_vis = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    response_color = cv2.applyColorMap(response_vis, cv2.COLORMAP_JET)

    cv2.imshow("Original", img)
    cv2.imshow("Harris Response", response_color)
    cv2.imshow("Detected Corners", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 示例9：多尺度Harris角点检测

```python
"""
示例9：多尺度Harris角点检测
在图像金字塔上进行角点检测
"""
import cv2
import numpy as np

def multi_scale_harris(img, num_scales=4, scale_factor=1.5):
    """
    多尺度Harris角点检测

    Args:
        img: 输入图像
        num_scales: 尺度数量
        scale_factor: 尺度因子

    Returns:
        all_corners: 所有尺度的角点列表
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    all_corners = []
    current_img = gray.copy()

    for scale in range(num_scales):
        # 当前尺度的block_size
        block_size = int(2 + scale)

        # Harris检测
        gray_float = current_img.astype(np.float32)
        response = cv2.cornerHarris(gray_float, block_size, 3, 0.04)

        # NMS
        dilated = cv2.dilate(response, None)
        local_max = (response == dilated)
        thresh = 0.01 * response.max() if response.max() > 0 else 0
        corners_mask = (response > thresh) & local_max

        # 获取坐标
        y, x = np.where(corners_mask)

        # 转换到原始尺度
        scale_ratio = scale_factor ** scale
        corners = [(int(xi * scale_ratio), int(yi * scale_ratio))
                   for xi, yi in zip(x, y)]

        all_corners.append({
            'scale': scale,
            'corners': corners,
            'block_size': block_size
        })

        # 下采样
        if scale < num_scales - 1:
            new_size = (int(current_img.shape[1] / scale_factor),
                       int(current_img.shape[0] / scale_factor))
            current_img = cv2.resize(current_img, new_size)

    return all_corners

# 创建测试图像（包含不同尺度的特征）
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (200, 200, 200)

# 小特征
cv2.rectangle(img, (30, 30), (60, 60), (100, 100, 100), -1)
cv2.rectangle(img, (80, 30), (110, 60), (100, 100, 100), -1)

# 中等特征
cv2.rectangle(img, (150, 50), (250, 150), (80, 80, 80), -1)

# 大特征
cv2.rectangle(img, (300, 50), (550, 300), (60, 60, 60), -1)

# 添加噪声
noise = np.random.normal(0, 3, img.shape).astype(np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# 多尺度检测
results = multi_scale_harris(img, num_scales=4, scale_factor=1.5)

print("多尺度Harris角点检测:")
print("-" * 50)

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
result_img = img.copy()

for i, scale_result in enumerate(results):
    scale = scale_result['scale']
    corners = scale_result['corners']
    block_size = scale_result['block_size']

    print(f"尺度 {scale} (block_size={block_size}): {len(corners)} 个角点")

    # 绘制角点（不同颜色表示不同尺度）
    for x, y in corners:
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            radius = 3 + scale * 2
            cv2.circle(result_img, (x, y), radius, colors[i], 2)

# 添加图例
legend_y = 20
for i, color in enumerate(colors):
    cv2.circle(result_img, (20, legend_y + i*25), 5, color, -1)
    cv2.putText(result_img, f"Scale {i}", (35, legend_y + i*25 + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cv2.imshow("Multi-scale Harris Corners", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例10：交互式Harris角点检测

```python
"""
示例10：交互式Harris角点检测
使用滑块调整参数
"""
import cv2
import numpy as np

class InteractiveHarrisDetector:
    """交互式Harris角点检测器"""

    def __init__(self, img):
        self.img = img
        if len(img.shape) == 3:
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = img.copy()

        self.gray_float = self.gray.astype(np.float32)

        # 参数
        self.block_size = 2
        self.k = 4  # k * 0.01
        self.threshold = 1  # threshold * 0.01

        self.window_name = "Interactive Harris Corner Detection"

    def update(self, val=None):
        """更新检测结果"""
        # 获取参数
        self.block_size = max(2, cv2.getTrackbarPos("Block Size", self.window_name))
        k = cv2.getTrackbarPos("k (x0.01)", self.window_name) * 0.01
        threshold = cv2.getTrackbarPos("Threshold (x0.01)", self.window_name) * 0.01

        # 确保k不为0
        k = max(0.01, k)
        threshold = max(0.001, threshold)

        # Harris检测
        response = cv2.cornerHarris(self.gray_float, self.block_size, 3, k)

        # NMS
        dilated = cv2.dilate(response, None)
        local_max = (response == dilated)
        thresh_value = threshold * response.max() if response.max() > 0 else 0
        corners_mask = (local_max) & (response > thresh_value)

        # 统计
        corner_count = np.sum(corners_mask)

        # 可视化
        result = self.img.copy() if len(self.img.shape) == 3 else cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        result[cv2.dilate(corners_mask.astype(np.uint8), None) > 0] = [0, 0, 255]

        # 添加信息
        info = [
            f"Block Size: {self.block_size}",
            f"k: {k:.3f}",
            f"Threshold: {threshold:.3f}",
            f"Corners: {corner_count}"
        ]

        for i, text in enumerate(info):
            cv2.putText(result, text, (10, 25 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示响应图
        response_vis = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        response_color = cv2.applyColorMap(response_vis, cv2.COLORMAP_JET)

        # 组合显示
        combined = np.hstack([result, response_color])
        cv2.imshow(self.window_name, combined)

    def run(self):
        """运行交互检测器"""
        cv2.namedWindow(self.window_name)

        # 创建滑块
        cv2.createTrackbar("Block Size", self.window_name, 2, 10, self.update)
        cv2.createTrackbar("k (x0.01)", self.window_name, 4, 20, self.update)
        cv2.createTrackbar("Threshold (x0.01)", self.window_name, 1, 10, self.update)

        # 初始显示
        self.update()

        print("调整滑块来改变检测参数")
        print("按 'q' 退出")

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    # 创建测试图像
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (220, 220, 220)

    # 添加形状
    cv2.rectangle(img, (50, 50), (200, 200), (100, 100, 100), -1)
    cv2.rectangle(img, (250, 80), (450, 300), (80, 80, 80), -1)
    cv2.circle(img, (550, 100), 40, (120, 120, 120), -1)

    pts = np.array([[100, 280], [180, 380], [50, 380]])
    cv2.fillPoly(img, [pts], (90, 90, 90))

    # 添加细节
    cv2.line(img, (300, 350), (550, 380), (70, 70, 70), 3)

    # 运行交互检测器
    detector = InteractiveHarrisDetector(img)
    detector.run()
```

## 3. Harris角点检测最佳实践

### 3.1 参数选择建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| blockSize | 2-5 | 小值检测细节，大值检测大特征 |
| ksize | 3 | Sobel算子大小，通常固定为3 |
| k | 0.04-0.06 | 过小会检测更多点，过大可能遗漏 |
| threshold | 0.01-0.1 | 相对于最大响应值 |

### 3.2 常见问题解决

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 角点过多 | 阈值太低 | 提高threshold或k |
| 角点过少 | 阈值太高 | 降低threshold或k |
| 边缘误检 | k值不当 | 调整k值 |
| 噪声影响 | 无预处理 | 先进行高斯模糊 |

## 4. 总结

本节详细介绍了Harris角点检测：

| 内容 | 要点 |
|------|------|
| 核心函数 | cv2.cornerHarris() |
| 响应函数 | R = det(M) - k·trace(M)² |
| 关键参数 | blockSize, k, threshold |
| 特性 | 旋转不变，对尺度敏感 |
| 后处理 | 非极大值抑制 |

## 5. 练习题

1. **基础练习**：
   - 实现完整的Harris角点检测流程
   - 分析k值对检测结果的影响

2. **进阶练习**：
   - 实现多尺度Harris角点检测
   - 比较Harris和其他角点检测算法

3. **实践项目**：
   - 使用Harris角点实现简单的图像配准
   - 创建角点跟踪器
