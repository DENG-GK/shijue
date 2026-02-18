"""
示例2：不同epsilon值的效果
- 精度从低到高对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((300, 300), dtype=np.uint8)
pts = np.array([[80, 40], [180, 20], [260, 70], [280, 160],
                [250, 250], [160, 280], [70, 250], [30, 160]])
cv2.fillPoly(img, [pts], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
perimeter = cv2.arcLength(cnt, True)

ratios = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('不同epsilon值效果', fontsize=14, fontweight='bold')

for idx, ratio in enumerate(ratios):
    epsilon = ratio * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.drawContours(canvas, [cnt], 0, (60, 60, 60), 1)
    cv2.drawContours(canvas, [approx], 0, (0, 255, 0), 2)
    for pt in approx:
        cv2.circle(canvas, tuple(pt[0]), 3, (0, 0, 255), -1)

    r, c = idx // 3, idx % 3
    axes[r, c].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[r, c].set_title(f'eps={ratio} → {len(approx)}点')
    axes[r, c].axis('off')

print(f"原始点数: {len(cnt)}")
for ratio in ratios:
    approx = cv2.approxPolyDP(cnt, ratio * perimeter, True)
    print(f"  eps={ratio}: {len(approx)}点")

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_epsilon_compare.png'), dpi=150, bbox_inches='tight')
plt.show()
