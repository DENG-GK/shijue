"""
示例5：形状因子
- 圆度、矩形度、长宽比、固实度、凸度
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 700), dtype=np.uint8)

# 圆形
cv2.circle(img, (80, 100), 60, 255, -1)
# 正方形
cv2.rectangle(img, (170, 40), (290, 160), 255, -1)
# 矩形
cv2.rectangle(img, (320, 50), (480, 150), 255, -1)
# 三角形
cv2.fillPoly(img, [np.array([[580, 40], [670, 160], [490, 160]])], 255)
# 星形
pts = []
for i in range(5):
    a_out = np.pi/2 + i * 2*np.pi/5
    a_in = np.pi/2 + (i + 0.5) * 2*np.pi/5
    pts.append([int(80 + 60*np.cos(a_out)), int(300 + 60*np.sin(a_out))])
    pts.append([int(80 + 25*np.cos(a_in)), int(300 + 25*np.sin(a_in))])
cv2.fillPoly(img, [np.array(pts)], 255)
# L形
cv2.rectangle(img, (180, 220), (240, 380), 255, -1)
cv2.rectangle(img, (240, 320), (340, 380), 255, -1)
# 椭圆
cv2.ellipse(img, (450, 300), (90, 45), 0, 0, 360, 255, -1)
# 不规则
cv2.fillPoly(img, [np.array([[560, 220], [670, 250], [650, 350], [580, 370], [530, 300]])], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_cnts = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1] // 200, cv2.boundingRect(c)[0]))
names = ["圆形", "正方形", "矩形", "三角形", "星形", "L形", "椭圆", "不规则"]

features = []
for i, cnt in enumerate(sorted_cnts):
    if i >= len(names):
        break
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    f = {
        'name': names[i],
        'circularity': 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0,
        'extent': area / (w * h) if w * h > 0 else 0,
        'aspect': w / h if h > 0 else 0,
        'solidity': area / hull_area if hull_area > 0 else 0,
    }
    features.append(f)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('形状因子分析', fontsize=14, fontweight='bold')

canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for i, cnt in enumerate(sorted_cnts):
    if i < len(names):
        cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(canvas, names[i], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
axes[0, 0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('形状标注')
axes[0, 0].axis('off')

# 各特征柱状图
factor_names = ['circularity', 'extent', 'aspect', 'solidity']
factor_labels = ['圆度', '矩形度', '长宽比', '固实度']
positions = [(0, 1), (0, 2), (1, 0), (1, 1)]

for (r, c), fname, flabel in zip(positions, factor_names, factor_labels):
    vals = [f[fname] for f in features]
    ns = [f['name'] for f in features]
    axes[r, c].barh(range(len(ns)), vals, color='steelblue', edgecolor='black')
    axes[r, c].set_yticks(range(len(ns)))
    axes[r, c].set_yticklabels(ns, fontsize=8)
    axes[r, c].set_xlabel(flabel)
    axes[r, c].set_title(flabel)

# 汇总表
info = f"{'形状':>6} {'圆度':>6} {'矩形度':>6} {'长宽比':>6} {'固实度':>6}\n"
info += "-" * 40 + "\n"
for f in features:
    info += f"{f['name']:>6} {f['circularity']:>6.3f} {f['extent']:>6.3f} {f['aspect']:>6.3f} {f['solidity']:>6.3f}\n"
axes[1, 2].text(0.05, 0.95, info, fontsize=8, family='monospace',
                verticalalignment='top', transform=axes[1, 2].transAxes)
axes[1, 2].axis('off')
axes[1, 2].set_title('汇总表')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_shape_factors.png'), dpi=150, bbox_inches='tight')
plt.show()
