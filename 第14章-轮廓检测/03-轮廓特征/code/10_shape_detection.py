"""
示例10：基于特征的形状检测
- ShapeDetector 类
- 多边形近似 + 形状因子综合判断
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ShapeDetector:
    """几何形状检测器"""

    def detect(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / h if h > 0 else 0
        extent = area / (w * h) if w * h > 0 else 0

        if circularity > 0.85:
            return "circle", circularity
        if vertices == 3:
            return "triangle", extent
        if vertices == 4:
            if 0.9 < aspect < 1.1 and extent > 0.85:
                return "square", extent
            return "rectangle", extent
        if vertices == 5:
            hull = cv2.convexHull(contour)
            solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            if solidity < 0.75:
                return "star", 1 - solidity
            return "pentagon", extent
        if vertices == 6:
            return "hexagon", extent
        if vertices > 6 and circularity > 0.7:
            return "ellipse", circularity
        return "polygon", 0.5


# 创建测试场景
img = np.ones((400, 700, 3), dtype=np.uint8) * 235

cv2.circle(img, (70, 100), 45, (80, 80, 80), -1)
cv2.rectangle(img, (150, 55), (250, 155), (80, 80, 80), -1)
cv2.rectangle(img, (280, 65), (400, 145), (80, 80, 80), -1)
cv2.fillPoly(img, [np.array([[490, 55], [560, 155], [420, 155]])], (80, 80, 80))
cv2.ellipse(img, (640, 105), (40, 55), 0, 0, 360, (80, 80, 80), -1)

# 下排
pts_p = []
for i in range(5):
    a = -np.pi/2 + i * 2*np.pi/5
    pts_p.append([int(70 + 45*np.cos(a)), int(290 + 45*np.sin(a))])
cv2.fillPoly(img, [np.array(pts_p)], (80, 80, 80))

pts_h = []
for i in range(6):
    a = i * np.pi/3
    pts_h.append([int(200 + 45*np.cos(a)), int(290 + 45*np.sin(a))])
cv2.fillPoly(img, [np.array(pts_h)], (80, 80, 80))

pts_s = []
for i in range(5):
    a_out = -np.pi/2 + i * 2*np.pi/5
    a_in = -np.pi/2 + (i + 0.5) * 2*np.pi/5
    pts_s.append([int(340 + 45*np.cos(a_out)), int(290 + 45*np.sin(a_out))])
    pts_s.append([int(340 + 20*np.cos(a_in)), int(290 + 20*np.sin(a_in))])
cv2.fillPoly(img, [np.array(pts_s)], (80, 80, 80))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

detector = ShapeDetector()
canvas = img.copy()

color_map = {
    'circle': (0, 200, 0), 'square': (200, 0, 0), 'rectangle': (0, 0, 200),
    'triangle': (200, 200, 0), 'ellipse': (200, 0, 200), 'pentagon': (0, 200, 200),
    'hexagon': (100, 200, 0), 'star': (200, 100, 0), 'polygon': (150, 150, 150),
}

for cnt in contours:
    if cv2.contourArea(cnt) < 100:
        continue
    shape, conf = detector.detect(cnt)
    color = color_map.get(shape, (200, 200, 200))
    cv2.drawContours(canvas, [cnt], 0, color, 2)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.putText(canvas, shape, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('基于特征的形状检测', fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('原始图像')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[1].set_title('检测结果')
axes[1].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_shape_detection.png'), dpi=150, bbox_inches='tight')
plt.show()
