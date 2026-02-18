"""
示例9：PerspectiveTransformer类
- 完整透视变换工具类
- 角点排序、自动计算尺寸
- 正/逆变换、点映射
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class PerspectiveTransformer:
    """完整的透视变换工具类"""

    def __init__(self):
        self.src_pts = None
        self.dst_pts = None
        self.H = None
        self.H_inv = None

    def set_source_points(self, points):
        """设置源角点（4个点）"""
        points = np.array(points, dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError("需要4个点")
        self.src_pts = self._order_points(points)

    def set_destination_rectangle(self, width=None, height=None, margin=0):
        """设置目标矩形"""
        if self.src_pts is None:
            raise ValueError("先设置源点")
        if width is None:
            w1 = np.linalg.norm(self.src_pts[0] - self.src_pts[1])
            w2 = np.linalg.norm(self.src_pts[2] - self.src_pts[3])
            width = int(max(w1, w2))
        if height is None:
            h1 = np.linalg.norm(self.src_pts[0] - self.src_pts[3])
            h2 = np.linalg.norm(self.src_pts[1] - self.src_pts[2])
            height = int(max(h1, h2))
        self.dst_pts = np.float32([
            [margin, margin], [width - margin, margin],
            [width - margin, height - margin], [margin, height - margin]
        ])
        self._compute_homography()
        return width, height

    def _order_points(self, pts):
        """排序：左上、右上、右下、左下"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _compute_homography(self):
        self.H = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.H_inv = np.linalg.inv(self.H)

    def warp(self, image, output_size=None):
        """正变换"""
        if self.H is None:
            raise ValueError("先计算变换")
        if output_size is None:
            output_size = (int(self.dst_pts[:, 0].max()),
                           int(self.dst_pts[:, 1].max()))
        return cv2.warpPerspective(image, self.H, output_size)

    def warp_inverse(self, image, output_size):
        """逆变换"""
        return cv2.warpPerspective(image, self.H_inv, output_size)

    def transform_point(self, point):
        """变换单个点"""
        pt = np.array([point[0], point[1], 1])
        t = self.H @ pt
        return t[:2] / t[2]

    def transform_points(self, points):
        """变换多个点"""
        return np.array([self.transform_point(p) for p in points])


# 演示
transformer = PerspectiveTransformer()

image = np.ones((400, 500, 3), dtype=np.uint8) * 200
cv2.rectangle(image, (80, 60), (420, 340), (255, 255, 255), -1)
cv2.putText(image, 'TRANSFORM', (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)

corners = [[80, 60], [420, 80], [400, 340], [100, 320]]
transformer.set_source_points(corners)
tw, th = transformer.set_destination_rectangle()
corrected = transformer.warp(image, (tw, th))

test_points = [[150, 150], [300, 200], [200, 280]]
transformed_points = transformer.transform_points(test_points)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('PerspectiveTransformer 类演示', fontsize=14, fontweight='bold')

img_marked = image.copy()
for pt in transformer.src_pts.astype(int):
    cv2.circle(img_marked, tuple(pt), 8, (0, 255, 0), -1)
for pt in test_points:
    cv2.circle(img_marked, tuple(pt), 5, (255, 0, 0), -1)
axes[0].imshow(cv2.cvtColor(img_marked, cv2.COLOR_BGR2RGB))
axes[0].set_title('源图像 (含测试点)')
axes[0].axis('off')

corrected_marked = corrected.copy()
for pt in transformer.dst_pts.astype(int):
    cv2.circle(corrected_marked, tuple(pt), 8, (0, 255, 0), -1)
for pt in transformed_points.astype(int):
    cv2.circle(corrected_marked, tuple(pt), 5, (255, 0, 0), -1)
axes[1].imshow(cv2.cvtColor(corrected_marked, cv2.COLOR_BGR2RGB))
axes[1].set_title('校正结果 (含变换后的点)')
axes[1].axis('off')

usage = f"""PerspectiveTransformer 用法:

t = PerspectiveTransformer()
t.set_source_points(corners)
w, h = t.set_destination_rectangle()
result = t.warp(image)
pts = t.transform_points(points)

输出尺寸: {tw}x{th}"""
axes[2].text(0.1, 0.5, usage, fontsize=10, family='monospace',
             verticalalignment='center', transform=axes[2].transAxes)
axes[2].axis('off')
axes[2].set_title('使用方法')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_perspective_class.png'), dpi=150, bbox_inches='tight')
plt.show()

print("PerspectiveTransformer 类演示完成！")
