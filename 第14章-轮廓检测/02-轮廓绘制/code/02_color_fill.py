"""
示例2：彩色填充
- 不同颜色填充轮廓
- 随机颜色映射
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
img = np.zeros((400, 500), dtype=np.uint8)

# 多个形状
cv2.rectangle(img, (20, 20), (150, 120), 255, -1)
cv2.circle(img, (250, 80), 55, 255, -1)
cv2.ellipse(img, (420, 80), (50, 35), 0, 0, 360, 255, -1)
pts = np.array([[60, 160], [180, 160], [150, 280], [30, 260]], np.int32)
cv2.fillPoly(img, [pts], 255)
cv2.circle(img, (280, 250), 65, 255, -1)
cv2.rectangle(img, (370, 180), (480, 350), 255, -1)
cv2.ellipse(img, (150, 350), (80, 30), 0, 0, 360, 255, -1)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('轮廓彩色填充', fontsize=14, fontweight='bold')

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始二值图')
axes[0].axis('off')

# 随机颜色填充
canvas1 = np.zeros((400, 500, 3), dtype=np.uint8)
for i, cnt in enumerate(contours):
    color = tuple(np.random.randint(50, 255, 3).tolist())
    cv2.drawContours(canvas1, [cnt], 0, color, -1)
axes[1].imshow(cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB))
axes[1].set_title('随机颜色填充')
axes[1].axis('off')

# 按面积着色（热力图）
canvas2 = np.zeros((400, 500, 3), dtype=np.uint8)
areas = [cv2.contourArea(cnt) for cnt in contours]
max_area = max(areas) if areas else 1
for i, cnt in enumerate(contours):
    ratio = areas[i] / max_area
    color = (int(255 * (1 - ratio)), int(255 * ratio), 0)  # 红→绿
    cv2.drawContours(canvas2, [cnt], 0, color, -1)
axes[2].imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))
axes[2].set_title('面积热力图 (红=小, 绿=大)')
axes[2].axis('off')

# 轮廓线 + 半透明填充
canvas3 = np.ones((400, 500, 3), dtype=np.uint8) * 240
overlay = canvas3.copy()
for i, cnt in enumerate(contours):
    color = tuple(np.random.randint(50, 200, 3).tolist())
    cv2.drawContours(overlay, [cnt], 0, color, -1)
cv2.addWeighted(overlay, 0.4, canvas3, 0.6, 0, canvas3)
for i, cnt in enumerate(contours):
    cv2.drawContours(canvas3, [cnt], 0, (0, 0, 0), 2)
axes[3].imshow(cv2.cvtColor(canvas3, cv2.COLOR_BGR2RGB))
axes[3].set_title('半透明填充 + 轮廓线')
axes[3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_color_fill.png'), dpi=150, bbox_inches='tight')
plt.show()
