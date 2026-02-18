"""
示例3：外接矩形
- boundingRect 直立外接矩形
- minAreaRect 最小外接矩形
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 600), dtype=np.uint8)
# 旋转矩形
center, size, angle = (150, 200), (150, 80), 30
rect_pts = cv2.boxPoints((center, size, angle))
cv2.fillPoly(img, [np.int32(rect_pts)], 255)
# 旋转椭圆
cv2.ellipse(img, (400, 200), (100, 50), -25, 0, 360, 255, -1)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('外接矩形对比', fontsize=14, fontweight='bold')

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

canvas1 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
canvas2 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

for cnt in contours:
    # 直立外接矩形
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(canvas1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(canvas1, f"{w}x{h}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 最小外接矩形
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    cv2.drawContours(canvas2, [np.int32(box)], 0, (0, 0, 255), 2)
    cv2.putText(canvas2, f"{int(rect[1][0])}x{int(rect[1][1])} {rect[2]:.1f}deg",
                (int(rect[0][0]) - 50, int(rect[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

axes[1].imshow(cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB))
axes[1].set_title('直立外接矩形')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))
axes[2].set_title('最小外接矩形')
axes[2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_bounding_rect.png'), dpi=150, bbox_inches='tight')
plt.show()
