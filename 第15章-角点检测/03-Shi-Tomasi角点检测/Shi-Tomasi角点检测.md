# Shi-Tomasi角点检测

## 1. Shi-Tomasi算法概述

Shi-Tomasi角点检测是由Jianbo Shi和Carlo Tomasi于1994年提出的改进版Harris角点检测算法。该算法在OpenCV中通过`cv2.goodFeaturesToTrack()`函数实现。

### 1.1 算法原理

Shi-Tomasi算法对Harris响应函数进行了改进：

**Harris响应**：
$$R = \lambda_1 \cdot \lambda_2 - k(\lambda_1 + \lambda_2)^2$$

**Shi-Tomasi响应**：
$$R = \min(\lambda_1, \lambda_2)$$

Shi-Tomasi直接使用较小的特征值作为角点响应，避免了Harris中k参数的选择问题。

### 1.2 与Harris的比较

| 特性 | Harris | Shi-Tomasi |
|------|--------|------------|
| 响应函数 | det(M) - k·trace(M)² | min(λ₁, λ₂) |
| 参数 | 需要调节k | 无需k参数 |
| 稳定性 | 较好 | 更好 |
| 速度 | 稍快 | 需计算特征值 |
| 应用 | 通用角点检测 | 特征跟踪 |

### 1.3 cv2.goodFeaturesToTrack() 函数

```python
corners = cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, ...)
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| image | ndarray | 输入灰度图像 |
| maxCorners | int | 返回的最大角点数 |
| qualityLevel | float | 质量水平（0-1） |
| minDistance | float | 角点间最小距离 |
| mask | ndarray | 检测区域掩码 |
| blockSize | int | 邻域大小 |
| useHarrisDetector | bool | 是否使用Harris |
| k | float | Harris参数 |

**返回值**：
- corners: 角点坐标数组，形状为(N, 1, 2)

## 2. 代码示例

### 示例1：基本Shi-Tomasi角点检测

```python
"""
示例1：基本Shi-Tomasi角点检测
使用goodFeaturesToTrack()检测角点
"""
import cv2
import numpy as np

# 创建测试图像
img = np.zeros((400, 500), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), 200, -1)
cv2.rectangle(img, (250, 80), (450, 300), 200, -1)
cv2.circle(img, (150, 320), 60, 200, -1)

# Shi-Tomasi角点检测
corners = cv2.goodFeaturesToTrack(
    img,
    maxCorners=50,      # 最多返回50个角点
    qualityLevel=0.01,  # 质量水平
    minDistance=10      # 角点间最小距离
)

print(f"检测到 {len(corners)} 个角点")

# 绘制角点
result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for corner in corners:
    x, y = corner.ravel().astype(int)
    cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

cv2.imshow("Original", img)
cv2.imshow("Shi-Tomasi Corners", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例2：参数影响分析

```python
"""
示例2：参数影响分析
分析maxCorners、qualityLevel、minDistance的影响
"""
import cv2
import numpy as np

# 创建测试图像
img = np.zeros((300, 400), dtype=np.uint8)
# 多个矩形创建多个角点
cv2.rectangle(img, (20, 20), (80, 80), 200, -1)
cv2.rectangle(img, (100, 20), (160, 80), 200, -1)
cv2.rectangle(img, (180, 20), (240, 80), 200, -1)
cv2.rectangle(img, (260, 20), (320, 80), 200, -1)
cv2.rectangle(img, (50, 120), (150, 220), 200, -1)
cv2.rectangle(img, (200, 120), (350, 250), 200, -1)

# 参数组合测试
param_sets = [
    {"maxCorners": 10, "qualityLevel": 0.01, "minDistance": 10, "name": "max=10"},
    {"maxCorners": 50, "qualityLevel": 0.01, "minDistance": 10, "name": "max=50"},
    {"maxCorners": 50, "qualityLevel": 0.1, "minDistance": 10, "name": "quality=0.1"},
    {"maxCorners": 50, "qualityLevel": 0.01, "minDistance": 30, "name": "dist=30"},
]

print("参数影响分析:")
print("-" * 50)
print(f"{'参数设置':>15} {'检测角点数':>12}")
print("-" * 50)

results = []
for params in param_sets:
    corners = cv2.goodFeaturesToTrack(
        img,
        maxCorners=params["maxCorners"],
        qualityLevel=params["qualityLevel"],
        minDistance=params["minDistance"]
    )

    count = len(corners) if corners is not None else 0
    print(f"{params['name']:>15} {count:>12}")

    # 可视化
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel().astype(int)
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1)

    cv2.putText(canvas, params["name"], (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(canvas, f"n={count}", (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    results.append(canvas)

# 组合显示
row1 = np.hstack(results[:2])
row2 = np.hstack(results[2:])
combined = np.vstack([row1, row2])

cv2.imshow("Parameter Analysis", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例3：使用掩码限制检测区域

```python
"""
示例3：使用掩码限制检测区域
只在指定区域内检测角点
"""
import cv2
import numpy as np

# 创建测试图像
img = np.zeros((400, 500), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), 200, -1)
cv2.rectangle(img, (250, 80), (450, 300), 200, -1)
cv2.circle(img, (150, 320), 60, 200, -1)

# 创建掩码 - 只检测左半边
mask = np.zeros(img.shape, dtype=np.uint8)
mask[:, :250] = 255

# 不使用掩码
corners_no_mask = cv2.goodFeaturesToTrack(
    img,
    maxCorners=50,
    qualityLevel=0.01,
    minDistance=10
)

# 使用掩码
corners_with_mask = cv2.goodFeaturesToTrack(
    img,
    maxCorners=50,
    qualityLevel=0.01,
    minDistance=10,
    mask=mask
)

print(f"无掩码: {len(corners_no_mask)} 个角点")
print(f"有掩码: {len(corners_with_mask)} 个角点")

# 可视化
result1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
result2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
result2[:, 250:] = result2[:, 250:] // 2  # 暗化掩码外区域

for corner in corners_no_mask:
    x, y = corner.ravel().astype(int)
    cv2.circle(result1, (x, y), 5, (0, 0, 255), -1)

for corner in corners_with_mask:
    x, y = corner.ravel().astype(int)
    cv2.circle(result2, (x, y), 5, (0, 0, 255), -1)

# 绘制掩码边界
cv2.line(result2, (250, 0), (250, 400), (0, 255, 255), 2)

cv2.putText(result1, "No Mask", (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(result2, "With Mask (left half)", (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

combined = np.hstack([result1, result2])
cv2.imshow("Mask Comparison", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例4：Harris vs Shi-Tomasi比较

```python
"""
示例4：Harris vs Shi-Tomasi比较
使用相同参数比较两种方法
"""
import cv2
import numpy as np

# 创建测试图像
img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (30, 30), (120, 120), 200, -1)
cv2.rectangle(img, (150, 50), (280, 180), 200, -1)
cv2.circle(img, (350, 100), 40, 200, -1)
cv2.fillPoly(img, [np.array([[80, 180], [160, 280], [30, 280]])], 200)

# Shi-Tomasi (默认)
corners_shi = cv2.goodFeaturesToTrack(
    img,
    maxCorners=30,
    qualityLevel=0.01,
    minDistance=10,
    useHarrisDetector=False  # 默认
)

# Harris (通过goodFeaturesToTrack)
corners_harris = cv2.goodFeaturesToTrack(
    img,
    maxCorners=30,
    qualityLevel=0.01,
    minDistance=10,
    useHarrisDetector=True,
    k=0.04
)

print("检测结果比较:")
print(f"Shi-Tomasi: {len(corners_shi) if corners_shi is not None else 0} 个角点")
print(f"Harris: {len(corners_harris) if corners_harris is not None else 0} 个角点")

# 可视化
result_shi = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
result_harris = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if corners_shi is not None:
    for corner in corners_shi:
        x, y = corner.ravel().astype(int)
        cv2.circle(result_shi, (x, y), 5, (0, 0, 255), -1)

if corners_harris is not None:
    for corner in corners_harris:
        x, y = corner.ravel().astype(int)
        cv2.circle(result_harris, (x, y), 5, (0, 255, 0), -1)

cv2.putText(result_shi, "Shi-Tomasi", (10, 25),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.putText(result_harris, "Harris", (10, 25),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 对比图 - 同时显示两种方法的结果
result_both = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
if corners_shi is not None:
    for corner in corners_shi:
        x, y = corner.ravel().astype(int)
        cv2.circle(result_both, (x, y), 6, (0, 0, 255), 2)
if corners_harris is not None:
    for corner in corners_harris:
        x, y = corner.ravel().astype(int)
        cv2.circle(result_both, (x, y), 4, (0, 255, 0), -1)

cv2.putText(result_both, "Red=Shi, Green=Harris", (10, 25),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

combined = np.hstack([result_shi, result_harris, result_both])
cv2.imshow("Comparison", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例5：角点质量评估

```python
"""
示例5：角点质量评估
分析检测到的角点的质量分布
"""
import cv2
import numpy as np

def compute_corner_quality(img, corners, block_size=3):
    """
    计算每个角点的质量（最小特征值）
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32)

    # 计算最小特征值图
    eigenvalues = cv2.cornerMinEigenVal(gray_float, block_size)

    qualities = []
    for corner in corners:
        x, y = corner.ravel().astype(int)
        if 0 <= y < eigenvalues.shape[0] and 0 <= x < eigenvalues.shape[1]:
            qualities.append(eigenvalues[y, x])
        else:
            qualities.append(0)

    return np.array(qualities)

# 创建测试图像
img = np.zeros((400, 500), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), 200, -1)
cv2.rectangle(img, (250, 80), (450, 300), 200, -1)
# 添加一些弱角点（模糊边缘）
blurred_rect = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(blurred_rect, (10, 10), (90, 90), 200, -1)
blurred_rect = cv2.GaussianBlur(blurred_rect, (15, 15), 0)
img[280:380, 50:150] = blurred_rect

# 检测角点
corners = cv2.goodFeaturesToTrack(
    img,
    maxCorners=50,
    qualityLevel=0.01,
    minDistance=10
)

# 计算质量
qualities = compute_corner_quality(img, corners)

print("角点质量分析:")
print("-" * 50)
print(f"检测到 {len(corners)} 个角点")
print(f"质量范围: [{qualities.min():.4f}, {qualities.max():.4f}]")
print(f"平均质量: {qualities.mean():.4f}")
print(f"质量标准差: {qualities.std():.4f}")

# 按质量分类角点
quality_threshold = qualities.max() * 0.3
high_quality = qualities > quality_threshold
low_quality = ~high_quality

print(f"\n高质量角点 (>{quality_threshold:.4f}): {np.sum(high_quality)}")
print(f"低质量角点: {np.sum(low_quality)}")

# 可视化 - 按质量着色
result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for i, corner in enumerate(corners):
    x, y = corner.ravel().astype(int)

    # 根据质量确定颜色（绿色=高质量，红色=低质量）
    quality_ratio = (qualities[i] - qualities.min()) / (qualities.max() - qualities.min() + 1e-10)
    color = (0, int(255 * quality_ratio), int(255 * (1 - quality_ratio)))

    cv2.circle(result, (x, y), 5, color, -1)
    # 标注质量值
    cv2.putText(result, f"{qualities[i]:.2f}", (x+5, y-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

cv2.putText(result, "Green=High Quality, Red=Low Quality", (10, 25),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 显示最小特征值图
eigenvalues = cv2.cornerMinEigenVal(img.astype(np.float32), 3)
eigen_vis = cv2.normalize(eigenvalues, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
eigen_color = cv2.applyColorMap(eigen_vis, cv2.COLORMAP_JET)

combined = np.hstack([result, eigen_color])
cv2.imshow("Corner Quality", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例6：特征跟踪应用

```python
"""
示例6：特征跟踪应用
使用Shi-Tomasi角点进行特征跟踪（Lucas-Kanade光流）
"""
import cv2
import numpy as np

class FeatureTracker:
    """特征点跟踪器"""

    def __init__(self, max_corners=100, quality_level=0.3, min_distance=7):
        # Shi-Tomasi参数
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )

        # Lucas-Kanade光流参数
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.prev_gray = None
        self.prev_points = None
        self.tracks = []
        self.track_len = 10

    def detect(self, gray):
        """检测新的特征点"""
        self.prev_gray = gray
        self.prev_points = cv2.goodFeaturesToTrack(gray, **self.feature_params)

        if self.prev_points is not None:
            self.tracks = [[tuple(p.ravel())] for p in self.prev_points]
        else:
            self.tracks = []

        return self.prev_points

    def track(self, gray):
        """跟踪特征点"""
        if self.prev_points is None or len(self.prev_points) == 0:
            return None

        # 计算光流
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )

        # 筛选有效点
        good_new = next_points[status.ravel() == 1]
        good_old = self.prev_points[status.ravel() == 1]

        # 更新轨迹
        new_tracks = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            idx = np.where(status.ravel() == 1)[0][i] if i < len(np.where(status.ravel() == 1)[0]) else -1
            if idx >= 0 and idx < len(self.tracks):
                track = self.tracks[idx]
                track.append(tuple(new.ravel()))
                if len(track) > self.track_len:
                    track.pop(0)
                new_tracks.append(track)

        self.tracks = new_tracks
        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)

        return good_new

    def draw(self, img):
        """绘制轨迹"""
        result = img.copy()

        # 绘制轨迹
        for track in self.tracks:
            if len(track) > 1:
                for i in range(len(track) - 1):
                    pt1 = tuple(map(int, track[i]))
                    pt2 = tuple(map(int, track[i+1]))
                    cv2.line(result, pt1, pt2, (0, 255, 0), 2)

            # 绘制当前点
            pt = tuple(map(int, track[-1]))
            cv2.circle(result, pt, 5, (0, 0, 255), -1)

        return result


# 模拟视频序列
def create_moving_scene(frame_num, total_frames=30):
    """创建移动场景"""
    img = np.ones((400, 500, 3), dtype=np.uint8) * 200

    # 移动的矩形
    offset = int(frame_num * 5)
    cv2.rectangle(img, (50 + offset, 100), (150 + offset, 200), (100, 100, 100), -1)

    # 固定的圆
    cv2.circle(img, (350, 200), 50, (80, 80, 80), -1)

    # 旋转的矩形
    angle = frame_num * 5
    center = (250, 300)
    pts = np.array([[-40, -40], [40, -40], [40, 40], [-40, 40]], dtype=float)
    rad = np.radians(angle)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    pts = (pts @ rot.T + center).astype(int)
    cv2.fillPoly(img, [pts], (120, 120, 120))

    return img


# 模拟跟踪
tracker = FeatureTracker(max_corners=50)
total_frames = 30

print("特征跟踪演示:")
print("按任意键播放下一帧，按 'q' 退出")

for frame_num in range(total_frames):
    # 创建帧
    frame = create_moving_scene(frame_num)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_num == 0:
        # 第一帧：检测特征
        points = tracker.detect(gray)
        print(f"检测到 {len(points) if points is not None else 0} 个特征点")
    else:
        # 后续帧：跟踪
        points = tracker.track(gray)
        if points is not None:
            print(f"帧 {frame_num}: 跟踪 {len(points)} 个点")

    # 绘制结果
    result = tracker.draw(frame)
    cv2.putText(result, f"Frame: {frame_num}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Feature Tracking", result)

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
```

### 示例7：角点检测器封装类

```python
"""
示例7：角点检测器封装类
封装Shi-Tomasi角点检测功能
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Corner:
    """角点数据类"""
    x: int
    y: int
    quality: float = 0.0

class ShiTomasiDetector:
    """Shi-Tomasi角点检测器"""

    def __init__(self,
                 max_corners: int = 100,
                 quality_level: float = 0.01,
                 min_distance: float = 10,
                 block_size: int = 3):
        """
        初始化检测器

        Args:
            max_corners: 最大角点数
            quality_level: 质量水平
            min_distance: 最小距离
            block_size: 邻域大小
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size

        self.corners: List[Corner] = []
        self.eigenvalue_map = None

    def detect(self,
               img: np.ndarray,
               mask: Optional[np.ndarray] = None) -> List[Corner]:
        """
        检测角点

        Args:
            img: 输入图像
            mask: 检测区域掩码

        Returns:
            角点列表
        """
        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # 检测角点
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=self.block_size
        )

        # 计算特征值图
        self.eigenvalue_map = cv2.cornerMinEigenVal(
            gray.astype(np.float32),
            self.block_size
        )

        # 转换为Corner对象
        self.corners = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel().astype(int)
                quality = self.eigenvalue_map[y, x] if (
                    0 <= y < self.eigenvalue_map.shape[0] and
                    0 <= x < self.eigenvalue_map.shape[1]
                ) else 0.0
                self.corners.append(Corner(x, y, quality))

        return self.corners

    def get_corner_coordinates(self) -> List[Tuple[int, int]]:
        """获取角点坐标列表"""
        return [(c.x, c.y) for c in self.corners]

    def get_numpy_points(self) -> np.ndarray:
        """获取numpy格式的点（用于光流等）"""
        if not self.corners:
            return None
        pts = np.array([[c.x, c.y] for c in self.corners], dtype=np.float32)
        return pts.reshape(-1, 1, 2)

    def filter_by_quality(self, min_quality: float) -> List[Corner]:
        """按质量筛选角点"""
        return [c for c in self.corners if c.quality >= min_quality]

    def filter_by_region(self,
                         x_range: Tuple[int, int],
                         y_range: Tuple[int, int]) -> List[Corner]:
        """按区域筛选角点"""
        return [c for c in self.corners
                if x_range[0] <= c.x <= x_range[1] and
                   y_range[0] <= c.y <= y_range[1]]

    def sort_by_quality(self, descending: bool = True) -> List[Corner]:
        """按质量排序"""
        return sorted(self.corners, key=lambda c: c.quality, reverse=descending)

    def visualize(self,
                  img: np.ndarray,
                  radius: int = 5,
                  color: Tuple[int, int, int] = (0, 0, 255),
                  show_quality: bool = False) -> np.ndarray:
        """可视化角点"""
        result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for corner in self.corners:
            cv2.circle(result, (corner.x, corner.y), radius, color, -1)

            if show_quality:
                cv2.putText(result, f"{corner.quality:.2f}",
                           (corner.x + 5, corner.y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return result

    def visualize_heatmap(self) -> np.ndarray:
        """可视化特征值热力图"""
        if self.eigenvalue_map is None:
            return None

        normalized = cv2.normalize(self.eigenvalue_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap


# 使用示例
if __name__ == "__main__":
    # 创建测试图像
    img = np.ones((400, 500, 3), dtype=np.uint8) * 220
    cv2.rectangle(img, (50, 50), (200, 200), (100, 100, 100), -1)
    cv2.rectangle(img, (250, 80), (450, 300), (80, 80, 80), -1)
    cv2.circle(img, (150, 320), 60, (120, 120, 120), -1)

    # 创建检测器
    detector = ShiTomasiDetector(max_corners=50, quality_level=0.01)

    # 检测
    corners = detector.detect(img)
    print(f"检测到 {len(corners)} 个角点")

    # 获取最高质量的5个角点
    top_corners = detector.sort_by_quality()[:5]
    print("\n质量最高的5个角点:")
    for c in top_corners:
        print(f"  ({c.x}, {c.y}): quality={c.quality:.4f}")

    # 可视化
    result = detector.visualize(img, show_quality=True)
    heatmap = detector.visualize_heatmap()

    cv2.imshow("Detected Corners", result)
    cv2.imshow("Eigenvalue Heatmap", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 示例8：自适应角点检测

```python
"""
示例8：自适应角点检测
根据图像内容自动调整参数
"""
import cv2
import numpy as np

def adaptive_corner_detection(img, target_corners=50, tolerance=10):
    """
    自适应角点检测

    Args:
        img: 输入图像
        target_corners: 目标角点数
        tolerance: 容差

    Returns:
        corners: 角点坐标
        params: 使用的参数
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # 初始参数
    quality_level = 0.01
    min_distance = 10

    # 二分搜索找合适的quality_level
    low, high = 0.001, 0.5
    best_corners = None
    best_quality = quality_level

    for _ in range(20):  # 最多迭代20次
        mid = (low + high) / 2

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=target_corners * 2,  # 允许更多，然后筛选
            qualityLevel=mid,
            minDistance=min_distance
        )

        count = len(corners) if corners is not None else 0

        if abs(count - target_corners) <= tolerance:
            best_corners = corners
            best_quality = mid
            break
        elif count < target_corners:
            high = mid
        else:
            low = mid

        if best_corners is None or abs(count - target_corners) < abs(len(best_corners) - target_corners):
            best_corners = corners
            best_quality = mid

    # 如果角点过多，只保留最强的
    if best_corners is not None and len(best_corners) > target_corners:
        # 计算每个角点的质量
        eigenvalues = cv2.cornerMinEigenVal(gray.astype(np.float32), 3)
        qualities = []
        for corner in best_corners:
            x, y = corner.ravel().astype(int)
            qualities.append(eigenvalues[y, x])

        # 按质量排序并保留前target_corners个
        indices = np.argsort(qualities)[::-1][:target_corners]
        best_corners = best_corners[indices]

    return best_corners, {"quality_level": best_quality, "min_distance": min_distance}

# 测试
img = np.ones((400, 600, 3), dtype=np.uint8) * 220

# 创建具有不同角点密度的区域
# 密集区域
for i in range(5):
    for j in range(5):
        cv2.rectangle(img, (20 + i*40, 20 + j*40), (50 + i*40, 50 + j*40), (100, 100, 100), -1)

# 稀疏区域
cv2.rectangle(img, (300, 100), (500, 300), (80, 80, 80), -1)

# 圆形
cv2.circle(img, (400, 350), 40, (120, 120, 120), -1)

# 测试不同目标数量
target_counts = [20, 50, 100]

print("自适应角点检测:")
print("-" * 50)

results = []
for target in target_counts:
    corners, params = adaptive_corner_detection(img, target_corners=target)
    actual = len(corners) if corners is not None else 0

    print(f"目标: {target}, 实际: {actual}, quality_level: {params['quality_level']:.4f}")

    # 可视化
    result = img.copy()
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel().astype(int)
            cv2.circle(result, (x, y), 4, (0, 0, 255), -1)

    cv2.putText(result, f"Target={target}, Actual={actual}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    results.append(result)

# 组合显示
combined = np.hstack(results)
cv2.imshow("Adaptive Corner Detection", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例9：网格化角点检测

```python
"""
示例9：网格化角点检测
确保角点在图像中均匀分布
"""
import cv2
import numpy as np

def grid_based_detection(img, grid_rows=4, grid_cols=4, corners_per_cell=3):
    """
    基于网格的角点检测

    Args:
        img: 输入图像
        grid_rows: 网格行数
        grid_cols: 网格列数
        corners_per_cell: 每个单元格的角点数

    Returns:
        所有角点的列表
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape

    cell_h = h // grid_rows
    cell_w = w // grid_cols

    all_corners = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            # 定义单元格区域
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w

            # 创建掩码
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            # 在该区域检测角点
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=corners_per_cell,
                qualityLevel=0.01,
                minDistance=5,
                mask=mask
            )

            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel().astype(int)
                    all_corners.append((x, y, row, col))  # 保存网格信息

    return all_corners, (grid_rows, grid_cols, cell_h, cell_w)

# 创建测试图像
img = np.ones((400, 600, 3), dtype=np.uint8) * 220

# 添加不均匀分布的特征
# 左上角密集
for i in range(3):
    for j in range(3):
        cv2.rectangle(img, (20 + i*50, 20 + j*50), (60 + i*50, 60 + j*50), (100, 100, 100), -1)

# 右下角稀疏
cv2.rectangle(img, (350, 250), (550, 380), (80, 80, 80), -1)

# 中间
cv2.circle(img, (300, 150), 40, (120, 120, 120), -1)

# 普通检测
normal_corners = cv2.goodFeaturesToTrack(
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    maxCorners=30,
    qualityLevel=0.01,
    minDistance=10
)

# 网格化检测
grid_corners, grid_info = grid_based_detection(img, grid_rows=4, grid_cols=4, corners_per_cell=3)

print("角点检测比较:")
print(f"普通检测: {len(normal_corners) if normal_corners is not None else 0} 个角点")
print(f"网格化检测: {len(grid_corners)} 个角点")

# 可视化
result_normal = img.copy()
result_grid = img.copy()

# 绘制普通检测结果
if normal_corners is not None:
    for corner in normal_corners:
        x, y = corner.ravel().astype(int)
        cv2.circle(result_normal, (x, y), 5, (0, 0, 255), -1)

# 绘制网格化检测结果
grid_rows, grid_cols, cell_h, cell_w = grid_info

# 绘制网格线
for i in range(1, grid_rows):
    cv2.line(result_grid, (0, i * cell_h), (result_grid.shape[1], i * cell_h), (200, 200, 200), 1)
for j in range(1, grid_cols):
    cv2.line(result_grid, (j * cell_w, 0), (j * cell_w, result_grid.shape[0]), (200, 200, 200), 1)

# 绘制角点
for x, y, row, col in grid_corners:
    cv2.circle(result_grid, (x, y), 5, (0, 0, 255), -1)

cv2.putText(result_normal, "Normal Detection", (10, 25),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.putText(result_grid, "Grid-based Detection", (10, 25),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

combined = np.hstack([result_normal, result_grid])
cv2.imshow("Detection Comparison", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例10：实时角点检测

```python
"""
示例10：实时角点检测
模拟实时视频的角点检测
"""
import cv2
import numpy as np
import time

class RealTimeCornerDetector:
    """实时角点检测器"""

    def __init__(self):
        self.max_corners = 100
        self.quality_level = 0.01
        self.min_distance = 10

        self.fps_history = []
        self.max_history = 30

    def update_params(self, max_corners=None, quality_level=None, min_distance=None):
        """更新参数"""
        if max_corners is not None:
            self.max_corners = max_corners
        if quality_level is not None:
            self.quality_level = quality_level
        if min_distance is not None:
            self.min_distance = min_distance

    def detect(self, frame):
        """检测角点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance
        )

        return corners

    def draw(self, frame, corners, draw_fps=True):
        """绘制结果"""
        result = frame.copy()

        if corners is not None:
            for corner in corners:
                x, y = corner.ravel().astype(int)
                cv2.circle(result, (x, y), 4, (0, 0, 255), -1)

        # 显示信息
        info = [
            f"Corners: {len(corners) if corners is not None else 0}",
            f"Max: {self.max_corners}",
            f"Quality: {self.quality_level:.3f}",
            f"Min Dist: {self.min_distance}"
        ]

        if draw_fps and self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            info.insert(0, f"FPS: {avg_fps:.1f}")

        for i, text in enumerate(info):
            cv2.putText(result, text, (10, 25 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return result

    def update_fps(self, fps):
        """更新FPS历史"""
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_history:
            self.fps_history.pop(0)


def create_animated_frame(frame_num):
    """创建动画帧"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 220

    # 移动的矩形
    x = (frame_num * 3) % 500
    cv2.rectangle(frame, (x, 100), (x + 80, 180), (100, 100, 100), -1)

    # 旋转的正方形
    center = (320, 300)
    angle = frame_num * 2
    pts = np.array([[-50, -50], [50, -50], [50, 50], [-50, 50]], dtype=float)
    rad = np.radians(angle)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    pts = (pts @ rot.T + center).astype(int)
    cv2.fillPoly(frame, [pts], (80, 80, 80))

    # 缩放的圆
    radius = 30 + int(20 * np.sin(frame_num * 0.1))
    cv2.circle(frame, (550, 350), radius, (120, 120, 120), -1)

    # 固定的参考形状
    cv2.rectangle(frame, (50, 350), (150, 450), (60, 60, 60), -1)

    return frame


# 模拟实时检测
detector = RealTimeCornerDetector()

print("实时角点检测演示")
print("按 'q' 退出")
print("按 '+' 增加最大角点数, '-' 减少")
print("按 'u' 增加质量阈值, 'd' 降低")

frame_num = 0
while True:
    start_time = time.time()

    # 创建帧
    frame = create_animated_frame(frame_num)

    # 检测
    corners = detector.detect(frame)

    # 计算FPS
    elapsed = time.time() - start_time
    fps = 1.0 / elapsed if elapsed > 0 else 0
    detector.update_fps(fps)

    # 绘制
    result = detector.draw(frame, corners)

    cv2.imshow("Real-time Corner Detection", result)

    # 键盘控制
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        detector.update_params(max_corners=min(500, detector.max_corners + 10))
    elif key == ord('-'):
        detector.update_params(max_corners=max(10, detector.max_corners - 10))
    elif key == ord('u'):
        detector.update_params(quality_level=min(0.5, detector.quality_level + 0.01))
    elif key == ord('d'):
        detector.update_params(quality_level=max(0.001, detector.quality_level - 0.01))

    frame_num += 1

cv2.destroyAllWindows()
```

## 3. 最佳实践

### 3.1 参数选择建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| maxCorners | 50-200 | 根据应用需求 |
| qualityLevel | 0.01-0.1 | 越高越严格 |
| minDistance | 5-20 | 避免角点过于密集 |
| blockSize | 3-7 | 特征尺度 |

### 3.2 应用场景选择

| 场景 | 推荐设置 |
|------|----------|
| 特征跟踪 | 高质量，中等数量 |
| 图像匹配 | 均匀分布，大数量 |
| 实时应用 | 低数量，快速检测 |

## 4. 总结

本节介绍了Shi-Tomasi角点检测：

| 内容 | 要点 |
|------|------|
| 核心函数 | cv2.goodFeaturesToTrack() |
| 响应函数 | min(λ₁, λ₂) |
| 优势 | 无需调节k参数，更稳定 |
| 应用 | 特征跟踪首选 |
| 关键参数 | maxCorners, qualityLevel, minDistance |

## 5. 练习题

1. **基础练习**：
   - 比较Shi-Tomasi和Harris在噪声图像上的表现
   - 实现基于掩码的多区域角点检测

2. **进阶练习**：
   - 实现自适应质量阈值的角点检测
   - 创建网格化的均匀分布角点检测器

3. **实践项目**：
   - 使用Shi-Tomasi实现简单的目标跟踪
   - 创建基于角点的图像拼接系统
