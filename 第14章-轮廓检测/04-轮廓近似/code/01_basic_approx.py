"""
示例1：基本多边形近似
- approxPolyDP (Douglas-Peucker)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 500), dtype=np.uint8)
cv2.circle(img, (150, 200), 100, 255, -1)
pts = np.array([[300, 80], [420, 100], [450, 180], [430, 280], [350, 320], [280, 280], [260, 180]])
cv2.fillPoly(img, [pts], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('基本多边形近似', fontsize=14, fontweight='bold')

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

for idx, cnt in enumerate(contours):
    perimeter = cv2.arcLength(cnt, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    print(f"轮廓{idx}: {len(cnt)}点 → {len(approx)}点")

canvas = np.zeros((400, 500, 3), dtype=np.uint8)
for cnt in contours:
    cv2.drawContours(canvas, [cnt], 0, (80, 80, 80), 1)
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    cv2.drawContours(canvas, [approx], 0, (0, 255, 0), 2)
    for pt in approx:
        cv2.circle(canvas, tuple(pt[0]), 4, (0, 0, 255), -1)
axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[1].set_title('epsilon=2%周长')
axes[1].axis('off')

canvas2 = np.zeros((400, 500, 3), dtype=np.uint8)
for cnt in contours:
    cv2.drawContours(canvas2, [cnt], 0, (80, 80, 80), 1)
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)
    cv2.drawContours(canvas2, [approx], 0, (0, 255, 0), 2)
    for pt in approx:
        cv2.circle(canvas2, tuple(pt[0]), 4, (0, 0, 255), -1)
axes[2].imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))
axes[2].set_title('epsilon=5%周长')
axes[2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_basic_approx.png'), dpi=150, bbox_inches='tight')
plt.show()
