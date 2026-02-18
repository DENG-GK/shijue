"""
示例4：凸包近似
- 凸包简化轮廓
- 面积与固实度
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 500), dtype=np.uint8)
# 星形
pts_star = []
for i in range(5):
    a_out = -np.pi/2 + i * 2*np.pi/5
    a_in = -np.pi/2 + (i + 0.5) * 2*np.pi/5
    pts_star.append([int(150 + 100*np.cos(a_out)), int(200 + 100*np.sin(a_out))])
    pts_star.append([int(150 + 40*np.cos(a_in)), int(200 + 40*np.sin(a_in))])
cv2.fillPoly(img, [np.array(pts_star)], 255)
# L形
cv2.rectangle(img, (300, 80), (360, 320), 255, -1)
cv2.rectangle(img, (360, 260), (460, 320), 255, -1)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('凸包近似', fontsize=14, fontweight='bold')

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

for idx, cnt in enumerate(contours):
    hull = cv2.convexHull(cnt)
    a1 = cv2.contourArea(cnt)
    a2 = cv2.contourArea(hull)

    canvas = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(canvas, [hull], 0, (255, 0, 0), 2)
    for pt in hull:
        cv2.circle(canvas, tuple(pt[0]), 4, (0, 0, 255), -1)

    axes[idx + 1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    solidity = a1 / a2 if a2 > 0 else 0
    axes[idx + 1].set_title(f'原{len(cnt)}点→凸包{len(hull)}点\n固实度: {solidity:.3f}')
    axes[idx + 1].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_convex_hull_approx.png'), dpi=150, bbox_inches='tight')
plt.show()
