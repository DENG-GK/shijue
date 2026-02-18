"""
示例7：外接矩形近似
- 直立外接矩形 vs 最小外接矩形
- 面积效率比较
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 600), dtype=np.uint8)

# 旋转矩形
params = [
    ((100, 150), (120, 60), 30),
    ((280, 150), (100, 50), 60),
    ((450, 150), (150, 40), -20),
]
for center, size, angle in params:
    rect = cv2.boxPoints((center, size, angle))
    cv2.fillPoly(img, [np.int32(rect)], 255)

# 旋转椭圆
cv2.ellipse(img, (100, 320), (80, 40), 45, 0, 360, 255, -1)
# 不规则多边形
pts = np.array([[250, 250], [350, 280], [380, 350], [300, 400], [220, 350]])
cv2.fillPoly(img, [pts], 255)
# 三角形
pts2 = np.array([[430, 280], [550, 320], [480, 400]])
cv2.fillPoly(img, [pts2], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

print("外接矩形比较:")
print("-" * 80)
print(f"{'轮廓':>4} {'直立矩形':>20} {'直立面积':>10} {'最小矩形角度':>12} {'最小面积':>10} {'效率':>8}")
print("-" * 80)

for i, cnt in enumerate(contours):
    cnt_area = cv2.contourArea(cnt)
    # 直立外接矩形
    x, y, w, h = cv2.boundingRect(cnt)
    upright_area = w * h
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 最小外接矩形
    rect = cv2.minAreaRect(cnt)
    min_area = rect[1][0] * rect[1][1]
    box = np.int32(cv2.boxPoints(rect))
    cv2.drawContours(canvas, [box], 0, (0, 255, 0), 2)

    efficiency = cnt_area / min_area * 100 if min_area > 0 else 0
    print(f"{i:>4} ({x:>3},{y:>3}) {w:>3}x{h:>3} {upright_area:>10} "
          f"{rect[2]:>11.1f}° {min_area:>10.0f} {efficiency:>7.1f}%")

    cv2.drawContours(canvas, [cnt], 0, (0, 0, 255), 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('外接矩形近似', fontsize=14, fontweight='bold')
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[1].set_title('蓝=直立矩形, 绿=最小矩形, 红=轮廓')
axes[1].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_bounding_rect_approx.png'), dpi=150, bbox_inches='tight')
plt.show()
