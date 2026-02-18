"""
示例7：极值点
- 最左、最右、最上、最下
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 600), dtype=np.uint8)
shapes = [
    np.array([[100, 100], [200, 50], [280, 120], [250, 200], [150, 220], [80, 180]]),
    np.array([[350, 80], [480, 100], [520, 180], [480, 280], [380, 300], [300, 200]]),
    np.array([[100, 280], [200, 250], [250, 350], [150, 380], [80, 340]]),
]
for s in shapes:
    cv2.fillPoly(img, [s], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('轮廓极值点', fontsize=14, fontweight='bold')

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for cnt in contours:
    cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)
    left = tuple(cnt[cnt[:, :, 0].argmin()][0])
    right = tuple(cnt[cnt[:, :, 0].argmax()][0])
    top = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])

    cv2.circle(canvas, left, 6, (255, 0, 0), -1)
    cv2.circle(canvas, right, 6, (0, 0, 255), -1)
    cv2.circle(canvas, top, 6, (255, 255, 0), -1)
    cv2.circle(canvas, bottom, 6, (255, 0, 255), -1)

axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[1].set_title('极值点 (蓝=左, 红=右, 黄=上, 紫=下)')
axes[1].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_extreme_points.png'), dpi=150, bbox_inches='tight')
plt.show()
