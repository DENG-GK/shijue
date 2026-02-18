"""
示例1：面积和周长
- contourArea / arcLength
- 理论值对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

img = np.zeros((400, 600), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
cv2.circle(img, (300, 100), 50, 255, -1)
cv2.rectangle(img, (420, 60), (570, 140), 255, -1)
cv2.ellipse(img, (100, 280), (60, 40), 0, 0, 360, 255, -1)
pts = np.array([[280, 220], [380, 250], [360, 350], [260, 340], [240, 280]])
cv2.fillPoly(img, [pts], 255)
cv2.fillPoly(img, [np.array([[500, 200], [580, 350], [420, 350]])], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('轮廓面积和周长', fontsize=14, fontweight='bold')

canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
names = []
areas = []
perimeters = []

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    areas.append(area)
    perimeters.append(perimeter)
    names.append(f'#{i}')

    cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(canvas, f"A:{int(area)}", (cx - 25, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(canvas, f"P:{int(perimeter)}", (cx - 25, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

axes[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[0].set_title('轮廓标注')
axes[0].axis('off')

x = range(len(names))
axes[1].bar(x, areas, color='steelblue', edgecolor='black')
axes[1].set_xticks(x)
axes[1].set_xticklabels(names)
axes[1].set_ylabel('面积')
axes[1].set_title('面积对比')

axes[2].bar(x, perimeters, color='coral', edgecolor='black')
axes[2].set_xticks(x)
axes[2].set_xticklabels(names)
axes[2].set_ylabel('周长')
axes[2].set_title('周长对比')

print(f"理论值: 正方形100x100: A=10000, P=400")
print(f"理论值: 圆r=50: A≈{np.pi*50**2:.0f}, P≈{2*np.pi*50:.0f}")

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_area_perimeter.png'), dpi=150, bbox_inches='tight')
plt.show()
