"""
示例9：凸包绘制
- 轮廓与凸包对比
- 凸缺陷可视化
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
    pts_star.append([int(130 + 90 * np.cos(a_out)), int(150 + 90 * np.sin(a_out))])
    pts_star.append([int(130 + 35 * np.cos(a_in)), int(150 + 35 * np.sin(a_in))])
cv2.fillPoly(img, [np.array(pts_star)], 255)

# L形
cv2.rectangle(img, (280, 40), (350, 300), 255, -1)
cv2.rectangle(img, (350, 230), (460, 300), 255, -1)

# 手形
hand = np.array([[100, 350], [120, 350], [130, 290], [140, 350],
                 [160, 350], [170, 300], [180, 350], [200, 350],
                 [200, 380], [100, 380]])
cv2.fillPoly(img, [hand], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('凸包与凸缺陷绘制', fontsize=14, fontweight='bold')

for idx, cnt in enumerate(contours):
    canvas = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_GRAY2BGR)
    # 原轮廓
    cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)
    # 凸包
    hull = cv2.convexHull(cnt)
    cv2.drawContours(canvas, [hull], 0, (255, 0, 0), 2)
    # 凸缺陷
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    try:
        defects = cv2.convexityDefects(cnt, hull_idx)
        if defects is not None:
            for d in defects:
                s, e, f, depth = d[0]
                if depth / 256.0 > 5:
                    far = tuple(cnt[f][0])
                    cv2.circle(canvas, far, 5, (0, 0, 255), -1)
    except cv2.error:
        pass

    axes[idx].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    area_cnt = cv2.contourArea(cnt)
    area_hull = cv2.contourArea(hull)
    solidity = area_cnt / area_hull if area_hull > 0 else 0
    axes[idx].set_title(f'固实度: {solidity:.3f}')
    axes[idx].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_convex_hull_draw.png'), dpi=150, bbox_inches='tight')
plt.show()
