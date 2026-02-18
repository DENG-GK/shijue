"""
示例7：外接形状绘制
- 外接矩形、最小矩形、最小圆、拟合椭圆
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 500), dtype=np.uint8)
# 不规则形状
pts1 = np.array([[80, 60], [180, 40], [220, 120], [200, 200], [100, 180], [50, 120]])
cv2.fillPoly(img, [pts1], 255)
# 旋转椭圆
cv2.ellipse(img, (380, 120), (80, 40), 30, 0, 360, 255, -1)
# 星形
pts2 = []
for i in range(5):
    a_out = -np.pi/2 + i * 2*np.pi/5
    a_in = -np.pi/2 + (i + 0.5) * 2*np.pi/5
    pts2.append([int(120 + 70 * np.cos(a_out)), int(320 + 70 * np.sin(a_out))])
    pts2.append([int(120 + 30 * np.cos(a_in)), int(320 + 30 * np.sin(a_in))])
cv2.fillPoly(img, [np.array(pts2)], 255)
# L形
cv2.rectangle(img, (280, 220), (340, 380), 255, -1)
cv2.rectangle(img, (340, 320), (460, 380), 255, -1)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('外接形状绘制', fontsize=14, fontweight='bold')

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

# 直立外接矩形
canvas1 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(canvas1, (x, y), (x + w, y + h), (0, 255, 0), 2)
axes[0, 1].imshow(cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('直立外接矩形')
axes[0, 1].axis('off')

# 最小外接矩形
canvas2 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(canvas2, [box], 0, (0, 0, 255), 2)
axes[0, 2].imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('最小外接矩形')
axes[0, 2].axis('off')

# 最小外接圆
canvas3 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for cnt in contours:
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    cv2.circle(canvas3, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
axes[1, 0].imshow(cv2.cvtColor(canvas3, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('最小外接圆')
axes[1, 0].axis('off')

# 拟合椭圆
canvas4 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for cnt in contours:
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(canvas4, ellipse, (255, 0, 255), 2)
axes[1, 1].imshow(cv2.cvtColor(canvas4, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('拟合椭圆')
axes[1, 1].axis('off')

# 全部叠加
canvas5 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(canvas5, (x, y), (x + w, y + h), (0, 255, 0), 1)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    cv2.drawContours(canvas5, [np.int32(box)], 0, (0, 0, 255), 1)
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    cv2.circle(canvas5, (int(cx), int(cy)), int(radius), (255, 0, 0), 1)
    if len(cnt) >= 5:
        cv2.ellipse(canvas5, cv2.fitEllipse(cnt), (255, 0, 255), 1)
axes[1, 2].imshow(cv2.cvtColor(canvas5, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('全部叠加')
axes[1, 2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_bounding_shapes.png'), dpi=150, bbox_inches='tight')
plt.show()
