"""
示例8：轮廓简化工具类
- ContourSimplifier 封装多种近似方法
- 自适应简化与方法比较
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class ApproximationResult:
    """近似结果"""
    method: str
    points: np.ndarray
    point_count: int
    area: float
    compression_ratio: float


class ContourSimplifier:
    """轮廓简化工具类"""

    def __init__(self, contour: np.ndarray):
        self.contour = contour
        self.original_points = len(contour)
        self.original_area = cv2.contourArea(contour)
        self.perimeter = cv2.arcLength(contour, True)

    def polygon_approx(self, epsilon_ratio: float = 0.02) -> ApproximationResult:
        """多边形近似"""
        epsilon = epsilon_ratio * self.perimeter
        approx = cv2.approxPolyDP(self.contour, epsilon, True)
        return ApproximationResult(
            method=f"Polygon(ε={epsilon_ratio})",
            points=approx,
            point_count=len(approx),
            area=cv2.contourArea(approx),
            compression_ratio=len(approx) / self.original_points
        )

    def convex_hull(self) -> ApproximationResult:
        """凸包近似"""
        hull = cv2.convexHull(self.contour)
        return ApproximationResult(
            method="Convex Hull",
            points=hull,
            point_count=len(hull),
            area=cv2.contourArea(hull),
            compression_ratio=len(hull) / self.original_points
        )

    def bounding_rect(self) -> ApproximationResult:
        """直立外接矩形"""
        x, y, w, h = cv2.boundingRect(self.contour)
        points = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
        return ApproximationResult(
            method="Bounding Rect",
            points=points,
            point_count=4,
            area=w * h,
            compression_ratio=4 / self.original_points
        )

    def min_area_rect(self) -> ApproximationResult:
        """最小外接矩形"""
        rect = cv2.minAreaRect(self.contour)
        box = cv2.boxPoints(rect)
        points = np.int32(box).reshape(-1, 1, 2)
        return ApproximationResult(
            method="Min Area Rect",
            points=points,
            point_count=4,
            area=rect[1][0] * rect[1][1],
            compression_ratio=4 / self.original_points
        )

    def min_enclosing_circle(self) -> Tuple[Tuple[float, float], float]:
        """最小外接圆"""
        return cv2.minEnclosingCircle(self.contour)

    def fit_ellipse(self) -> Optional[Tuple]:
        """拟合椭圆"""
        if len(self.contour) >= 5:
            return cv2.fitEllipse(self.contour)
        return None

    def adaptive_simplify(self, target_points: int) -> ApproximationResult:
        """自适应简化到目标点数"""
        low, high = 0.001, 0.5
        best_approx = None
        best_diff = float('inf')

        for _ in range(50):
            mid = (low + high) / 2
            epsilon = mid * self.perimeter
            approx = cv2.approxPolyDP(self.contour, epsilon, True)
            diff = abs(len(approx) - target_points)
            if diff < best_diff:
                best_diff = diff
                best_approx = approx
            if len(approx) > target_points:
                low = mid
            else:
                high = mid

        return ApproximationResult(
            method=f"Adaptive({target_points}pts)",
            points=best_approx,
            point_count=len(best_approx),
            area=cv2.contourArea(best_approx),
            compression_ratio=len(best_approx) / self.original_points
        )

    def compare_all(self) -> List[ApproximationResult]:
        """比较所有近似方法"""
        return [
            self.polygon_approx(0.01),
            self.polygon_approx(0.02),
            self.polygon_approx(0.05),
            self.convex_hull(),
            self.bounding_rect(),
            self.min_area_rect(),
        ]


if __name__ == "__main__":
    # 创建复杂形状
    img = np.zeros((300, 300), dtype=np.uint8)
    pts = np.array([[50, 100], [150, 30], [250, 80], [280, 180],
                    [220, 270], [120, 280], [30, 220], [40, 150]])
    cv2.fillPoly(img, [pts], 255)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    simplifier = ContourSimplifier(cnt)

    # 打印比较
    print("轮廓近似方法比较:")
    print("=" * 80)
    print(f"原始轮廓: {simplifier.original_points} 点, 面积 {simplifier.original_area:.0f}")
    print("-" * 80)
    print(f"{'方法':>25} {'点数':>8} {'面积':>12} {'压缩比':>12} {'面积保留':>12}")
    print("-" * 80)

    results = simplifier.compare_all()
    for r in results:
        area_ratio = r.area / simplifier.original_area * 100 if simplifier.original_area > 0 else 0
        print(f"{r.method:>25} {r.point_count:>8} {r.area:>12.0f} "
              f"{r.compression_ratio:>11.2%} {area_ratio:>11.1f}%")

    # 自适应简化
    adaptive = simplifier.adaptive_simplify(6)
    a_ratio = adaptive.area / simplifier.original_area * 100 if simplifier.original_area > 0 else 0
    print(f"{adaptive.method:>25} {adaptive.point_count:>8} {adaptive.area:>12.0f} "
          f"{adaptive.compression_ratio:>11.2%} {a_ratio:>11.1f}%")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('轮廓简化工具类 - 方法比较', fontsize=14, fontweight='bold')

    for idx, r in enumerate(results):
        row, col = idx // 3, idx % 3
        cell = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cell, [cnt], 0, (100, 100, 100), 1)
        cv2.drawContours(cell, [r.points], 0, (0, 255, 0), 2)
        for pt in r.points:
            cv2.circle(cell, tuple(pt[0]), 3, (0, 0, 255), -1)

        axes[row, col].imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
        area_ratio = r.area / simplifier.original_area * 100 if simplifier.original_area > 0 else 0
        axes[row, col].set_title(f'{r.method}\n{r.point_count}点, 面积{area_ratio:.1f}%')
        axes[row, col].axis('off')

    plt.tight_layout()
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '08_simplifier_class.png'), dpi=150, bbox_inches='tight')
    plt.show()
