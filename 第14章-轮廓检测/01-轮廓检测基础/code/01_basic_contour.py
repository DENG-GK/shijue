"""
示例1：基本轮廓检测
- findContours 基本用法
- 二值化 + 轮廓检测
- 轮廓数据结构
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:] = (240, 240, 240)

# 绘制多个形状
cv2.rectangle(img, (30, 30), (130, 130), (0, 0, 0), -1)
cv2.circle(img, (220, 80), 50, (0, 0, 0), -1)
cv2.ellipse(img, (340, 80), (40, 30), 0, 0, 360, (0, 0, 0), -1)
pts = np.array([[80, 180], [160, 180], [140, 270], [60, 270]], np.int32)
cv2.fillPoly(img, [pts], (0, 0, 0))
cv2.circle(img, (260, 230), 40, (0, 0, 0), -1)

# 转灰度 + 二值化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 查找轮廓
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"检测到 {len(contours)} 个轮廓")
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print(f"  轮廓{i}: 点数={len(cnt)}, 面积={area:.0f}, 周长={perimeter:.1f}")

# 可视化
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle('基本轮廓检测', fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('原始图像')
axes[0].axis('off')

axes[1].imshow(gray, cmap='gray')
axes[1].set_title('灰度图像')
axes[1].axis('off')

axes[2].imshow(binary, cmap='gray')
axes[2].set_title('二值图像')
axes[2].axis('off')

# 绘制轮廓
canvas = np.ones_like(img) * 255
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0), (128, 0, 255)]
for i, cnt in enumerate(contours):
    cv2.drawContours(canvas, [cnt], 0, colors[i % len(colors)], 2)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(canvas, str(i), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % len(colors)], 2)

axes[3].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[3].set_title(f'检测到 {len(contours)} 个轮廓')
axes[3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_basic_contour.png'), dpi=150, bbox_inches='tight')
plt.show()
