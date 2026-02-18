"""
示例9：轮廓特征分析器
- ContourAnalyzer 类
- 综合特征计算 + 形状分类
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ContourAnalyzer:
    """轮廓特征分析器"""

    def analyze(self, contour):
        """分析轮廓特征"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else 0
        cy = M["m01"] / M["m00"] if M["m00"] != 0 else 0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        return {
            'area': area, 'perimeter': perimeter,
            'bbox': (x, y, w, h), 'centroid': (cx, cy),
            'aspect_ratio': w / h if h > 0 else 0,
            'extent': area / (w * h) if w * h > 0 else 0,
            'circularity': 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0,
            'solidity': area / hull_area if hull_area > 0 else 0,
            'is_convex': cv2.isContourConvex(contour),
        }

    def classify(self, features):
        """简单形状分类"""
        if features['circularity'] > 0.85:
            return "圆形"
        elif features['circularity'] > 0.75 and 0.9 < features['aspect_ratio'] < 1.1:
            return "正方形"
        elif features['extent'] > 0.85:
            return "矩形"
        elif features['solidity'] < 0.7:
            return "星形/凹"
        elif 0.4 < features['extent'] < 0.6:
            return "三角形"
        else:
            return "不规则"


# 演示
img = np.zeros((300, 600), dtype=np.uint8)
cv2.circle(img, (80, 150), 60, 255, -1)
cv2.rectangle(img, (170, 90), (280, 210), 255, -1)
cv2.ellipse(img, (380, 150), (80, 40), 20, 0, 360, 255, -1)
pts = []
for i in range(5):
    a_out = -np.pi/2 + i * 2*np.pi/5
    a_in = -np.pi/2 + (i + 0.5) * 2*np.pi/5
    pts.append([int(540 + 55*np.cos(a_out)), int(150 + 55*np.sin(a_out))])
    pts.append([int(540 + 22*np.cos(a_in)), int(150 + 22*np.sin(a_in))])
cv2.fillPoly(img, [np.array(pts)], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_cnts = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

analyzer = ContourAnalyzer()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('ContourAnalyzer 类演示', fontsize=14, fontweight='bold')

canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
info = ""
for i, cnt in enumerate(sorted_cnts):
    features = analyzer.analyze(cnt)
    shape = analyzer.classify(features)
    cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)
    x, y, w, h = features['bbox']
    cv2.putText(canvas, shape, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    info += f"{shape}: C={features['circularity']:.3f} E={features['extent']:.3f} S={features['solidity']:.3f}\n"

axes[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[0].set_title('形状分类')
axes[0].axis('off')

axes[1].text(0.05, 0.5, info, fontsize=11, family='monospace',
             verticalalignment='center', transform=axes[1].transAxes)
axes[1].axis('off')
axes[1].set_title('特征值')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_contour_analyzer.png'), dpi=150, bbox_inches='tight')
plt.show()
