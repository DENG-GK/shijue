"""
示例5：直线拟合
- cv2.fitLine 拟合最佳直线
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 500), dtype=np.uint8)
# 倾斜矩形
center1, size1, angle1 = (150, 150), (200, 30), 30
rect1 = cv2.boxPoints((center1, size1, angle1))
cv2.fillPoly(img, [np.int32(rect1)], 255)
# 另一个
center2, size2, angle2 = (350, 250), (180, 25), -45
rect2 = cv2.boxPoints((center2, size2, angle2))
cv2.fillPoly(img, [np.int32(rect2)], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

for cnt in contours:
    if len(cnt) < 5:
        continue
    line = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    angle = np.degrees(np.arctan2(vy, vx))

    cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)
    lefty = int((-x0 * vy / vx) + y0) if vx != 0 else 0
    righty = int(((500 - x0) * vy / vx) + y0) if vx != 0 else 0
    cv2.line(canvas, (0, lefty), (500, righty), (0, 0, 255), 2)
    cv2.circle(canvas, (int(x0), int(y0)), 5, (255, 0, 0), -1)
    print(f"方向: ({vx:.3f}, {vy:.3f}), 角度: {angle:.1f}°")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('直线拟合 (fitLine)', fontsize=14, fontweight='bold')
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[1].set_title('拟合直线 (红=直线, 蓝=通过点)')
axes[1].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_fit_line.png'), dpi=150, bbox_inches='tight')
plt.show()
