"""
示例8：点到轮廓的距离
- pointPolygonTest
- 距离图可视化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((300, 400), dtype=np.uint8)
pts = np.array([[120, 60], [300, 60], [350, 150], [300, 240], [120, 240], [70, 150]])
cv2.fillPoly(img, [pts], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

# 计算距离图
h, w = img.shape
dist_map = np.zeros((h, w), dtype=np.float32)
for y in range(h):
    for x in range(w):
        dist_map[y, x] = cv2.pointPolygonTest(cnt, (float(x), float(y)), True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('点到轮廓距离 (pointPolygonTest)', fontsize=14, fontweight='bold')

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

# 距离热力图
im = axes[1].imshow(dist_map, cmap='RdBu', vmin=-100, vmax=100)
plt.colorbar(im, ax=axes[1], fraction=0.046)
axes[1].set_title('距离图 (蓝=内, 红=外)')
axes[1].axis('off')

# 测试点
test_points = [(200, 150), (120, 60), (30, 30), (250, 150)]
canvas = np.zeros((h, w, 3), dtype=np.uint8)
pos = np.maximum(dist_map, 0)
neg = np.maximum(-dist_map, 0)
max_p = pos.max() if pos.max() > 0 else 1
max_n = neg.max() if neg.max() > 0 else 1
canvas[:, :, 0] = (pos / max_p * 255).astype(np.uint8)
canvas[:, :, 2] = (neg / max_n * 255).astype(np.uint8)
canvas[np.abs(dist_map) < 1] = [255, 255, 255]

for pt in test_points:
    d = cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), True)
    color = (0, 255, 0) if d >= 0 else (0, 0, 255)
    cv2.circle(canvas, pt, 5, color, -1)
    cv2.putText(canvas, f"{d:.0f}", (pt[0] + 8, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

axes[2].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[2].set_title('距离可视化 + 测试点')
axes[2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_point_polygon_test.png'), dpi=150, bbox_inches='tight')
plt.show()
