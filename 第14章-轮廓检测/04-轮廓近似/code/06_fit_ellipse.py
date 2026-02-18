"""
示例6：椭圆拟合
- cv2.fitEllipse 拟合最佳椭圆
- 拟合误差计算
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 600), dtype=np.uint8)
# 椭圆
cv2.ellipse(img, (100, 150), (60, 40), 20, 0, 360, 255, -1)
# 圆形
cv2.circle(img, (280, 150), 50, 255, -1)
# 不规则形状
pts = np.array([[400, 80], [500, 100], [530, 180], [480, 240],
                [400, 250], [330, 200], [340, 120]])
cv2.fillPoly(img, [pts], 255)
# 长条形
center_r, size_r, angle_r = (150, 320), (150, 40), 25
rect = cv2.boxPoints((center_r, size_r, angle_r))
cv2.fillPoly(img, [np.int32(rect)], 255)
# 近似圆的多边形
np.random.seed(42)
pts2 = []
for i in range(8):
    a = i * 2 * np.pi / 8
    r = 45 + np.random.randint(-5, 5)
    pts2.append([int(420 + r * np.cos(a)), int(320 + r * np.sin(a))])
cv2.fillPoly(img, [np.array(pts2)], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

print("椭圆拟合结果:")
print("-" * 70)
print(f"{'轮廓':>4} {'中心':>15} {'长轴':>8} {'短轴':>8} {'角度':>8} {'拟合误差':>10}")
print("-" * 70)

for i, cnt in enumerate(contours):
    if len(cnt) < 5:
        continue
    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), (ma, MA), angle = ellipse

    cnt_area = cv2.contourArea(cnt)
    ell_area = np.pi * ma * MA / 4
    fit_error = abs(cnt_area - ell_area) / cnt_area * 100 if cnt_area > 0 else 0

    print(f"{i:>4} ({cx:>5.1f},{cy:>5.1f}) {ma:>8.1f} {MA:>8.1f} {angle:>7.1f}° {fit_error:>9.1f}%")

    cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)
    cv2.ellipse(canvas, ellipse, (255, 0, 0), 2)
    cv2.circle(canvas, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    # 主轴方向线
    length = MA / 2
    rad = np.radians(angle)
    x2 = int(cx + length * np.cos(rad))
    y2 = int(cy + length * np.sin(rad))
    cv2.line(canvas, (int(cx), int(cy)), (x2, y2), (0, 255, 255), 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('椭圆拟合 (fitEllipse)', fontsize=14, fontweight='bold')
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[1].set_title('拟合椭圆 (蓝=椭圆, 红=中心, 黄=主轴)')
axes[1].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_fit_ellipse.png'), dpi=150, bbox_inches='tight')
plt.show()
