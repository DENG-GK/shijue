"""
示例8：轮廓标注
- 标注编号、面积、质心
- 信息丰富的可视化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
img = np.zeros((400, 500), dtype=np.uint8)
cv2.rectangle(img, (20, 20), (160, 140), 255, -1)
cv2.circle(img, (280, 80), 60, 255, -1)
cv2.ellipse(img, (420, 100), (50, 35), 0, 0, 360, 255, -1)
pts = np.array([[60, 200], [180, 180], [200, 300], [80, 320], [30, 260]])
cv2.fillPoly(img, [pts], 255)
cv2.circle(img, (320, 280), 70, 255, -1)
cv2.rectangle(img, (400, 220), (480, 370), 255, -1)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('轮廓标注', fontsize=14, fontweight='bold')

# 编号标注
canvas1 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for i, cnt in enumerate(contours):
    cv2.drawContours(canvas1, [cnt], 0, (0, 255, 0), 2)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(canvas1, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(canvas1, f"#{i}", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
axes[0].imshow(cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB))
axes[0].set_title('编号+质心')
axes[0].axis('off')

# 面积标注
canvas2 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    cv2.drawContours(canvas2, [cnt], 0, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.putText(canvas2, f"A={int(area)}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
axes[1].imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))
axes[1].set_title('面积标注')
axes[1].axis('off')

# 综合标注
canvas3 = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    cv2.drawContours(canvas3, [cnt], 0, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(canvas3, (x, y), (x + w, y + h), (255, 0, 0), 1)

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(canvas3, (cx, cy), 3, (0, 0, 255), -1)

    label = f"#{i} C={circularity:.2f}"
    cv2.putText(canvas3, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
axes[2].imshow(cv2.cvtColor(canvas3, cv2.COLOR_BGR2RGB))
axes[2].set_title('综合标注')
axes[2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_label_contours.png'), dpi=150, bbox_inches='tight')
plt.show()
