"""
示例4：最小外接圆和拟合椭圆
- minEnclosingCircle
- fitEllipse
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 600), dtype=np.uint8)
pts1 = np.array([[50, 100], [150, 50], [200, 150], [150, 250], [50, 200]])
cv2.fillPoly(img, [pts1], 255)
pts2 = np.array([[300, 80], [420, 100], [450, 200], [400, 280], [280, 250], [260, 150]])
cv2.fillPoly(img, [pts2], 255)
cv2.ellipse(img, (150, 350), (100, 20), 20, 0, 360, 255, -1)
cv2.circle(img, (450, 330), 50, 255, -1)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('最小外接圆与拟合椭圆', fontsize=14, fontweight='bold')

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

canvas_c = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
canvas_e = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

for cnt in contours:
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    cv2.circle(canvas_c, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
    cv2.circle(canvas_c, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(canvas_e, ellipse, (255, 0, 0), 2)

axes[1].imshow(cv2.cvtColor(canvas_c, cv2.COLOR_BGR2RGB))
axes[1].set_title('最小外接圆')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(canvas_e, cv2.COLOR_BGR2RGB))
axes[2].set_title('拟合椭圆')
axes[2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_circle_ellipse.png'), dpi=150, bbox_inches='tight')
plt.show()
