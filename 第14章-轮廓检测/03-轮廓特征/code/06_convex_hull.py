"""
示例6：凸包和凸缺陷
- convexHull / convexityDefects
- isContourConvex
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 400), dtype=np.uint8)
palm = np.array([[160, 350], [240, 350], [260, 300], [270, 250],
                 [250, 150], [230, 80], [210, 80], [210, 150],
                 [190, 80], [170, 80], [170, 150],
                 [150, 100], [130, 100], [130, 170],
                 [130, 250], [120, 300]])
cv2.fillPoly(img, [palm], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

hull = cv2.convexHull(cnt)
hull_idx = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull_idx)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('凸包和凸缺陷', fontsize=14, fontweight='bold')

# 原轮廓
canvas1 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
cv2.drawContours(canvas1, [cnt], 0, (0, 255, 0), 2)
axes[0].imshow(cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'原轮廓 ({len(cnt)}点)')
axes[0].axis('off')

# 凸包
canvas2 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
cv2.drawContours(canvas2, [cnt], 0, (0, 255, 0), 1)
cv2.drawContours(canvas2, [hull], 0, (255, 0, 0), 2)
axes[1].imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))
is_convex = cv2.isContourConvex(cnt)
axes[1].set_title(f'凸包 ({len(hull)}点), 凸={is_convex}')
axes[1].axis('off')

# 凸缺陷
canvas3 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
cv2.drawContours(canvas3, [cnt], 0, (0, 255, 0), 1)
cv2.drawContours(canvas3, [hull], 0, (255, 0, 0), 2)
n_deep = 0
if defects is not None:
    for d in defects:
        s, e, f, depth = d[0]
        if depth / 256.0 > 10:
            n_deep += 1
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.circle(canvas3, far, 5, (0, 0, 255), -1)
            cv2.line(canvas3, start, far, (0, 255, 255), 1)
            cv2.line(canvas3, end, far, (0, 255, 255), 1)
axes[2].imshow(cv2.cvtColor(canvas3, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'凸缺陷 (深度>10: {n_deep}个)')
axes[2].axis('off')

print(f"凸包点数: {len(hull)}")
print(f"是否凸轮廓: {is_convex}")
print(f"凸缺陷数: {len(defects) if defects is not None else 0}")

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_convex_hull.png'), dpi=150, bbox_inches='tight')
plt.show()
