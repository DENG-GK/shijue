"""
示例3：使用近似识别几何形状
- 基于顶点数判断
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def detect_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    vertices = len(approx)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    if vertices == 3:
        return "Triangle", approx
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        if 0.95 <= w / float(h) <= 1.05:
            return "Square", approx
        return "Rectangle", approx
    elif vertices == 5:
        return "Pentagon", approx
    elif vertices == 6:
        return "Hexagon", approx
    elif circularity > 0.8:
        return "Circle", approx
    return "Ellipse", approx


img = np.ones((400, 700, 3), dtype=np.uint8) * 235

cv2.fillPoly(img, [np.array([[80, 40], [160, 160], [0, 160]])], (80, 80, 80))
cv2.rectangle(img, (200, 40), (320, 160), (80, 80, 80), -1)
cv2.rectangle(img, (360, 50), (500, 150), (80, 80, 80), -1)
pts_p = []
for i in range(5):
    a = -np.pi/2 + i * 2*np.pi/5
    pts_p.append([int(590 + 55*np.cos(a)), int(100 + 55*np.sin(a))])
cv2.fillPoly(img, [np.array(pts_p)], (80, 80, 80))
cv2.circle(img, (80, 290), 55, (80, 80, 80), -1)
cv2.ellipse(img, (240, 290), (75, 45), 0, 0, 360, (80, 80, 80), -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

canvas = img.copy()
colors = {'Triangle': (0, 200, 0), 'Square': (200, 0, 0), 'Rectangle': (0, 0, 200),
          'Pentagon': (200, 200, 0), 'Hexagon': (200, 0, 200),
          'Circle': (0, 200, 200), 'Ellipse': (100, 200, 100)}

for cnt in contours:
    if cv2.contourArea(cnt) < 500:
        continue
    name, approx = detect_shape(cnt)
    color = colors.get(name, (150, 150, 150))
    cv2.drawContours(canvas, [cnt], 0, color, 2)
    for pt in approx:
        cv2.circle(canvas, tuple(pt[0]), 4, (0, 0, 0), -1)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.putText(canvas, f"{name}({len(approx)})", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('多边形近似形状识别', fontsize=14, fontweight='bold')
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('原始图像')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[1].set_title('识别结果')
axes[1].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_shape_recognition.png'), dpi=150, bbox_inches='tight')
plt.show()
