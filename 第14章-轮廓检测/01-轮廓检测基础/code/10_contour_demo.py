"""
示例10：轮廓检测综合演示
- 模拟场景：多形状检测
- 统计信息汇总
- 适用于教学演示
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建模拟场景
np.random.seed(42)
img = np.ones((400, 600, 3), dtype=np.uint8) * 230

# 添加多种形状
cv2.rectangle(img, (30, 30), (160, 120), (60, 60, 60), -1)
cv2.circle(img, (260, 80), 50, (60, 60, 60), -1)
cv2.ellipse(img, (430, 80), (70, 40), 15, 0, 360, (60, 60, 60), -1)
pts1 = np.array([[80, 160], [180, 160], [150, 260], [50, 260]], np.int32)
cv2.fillPoly(img, [pts1], (60, 60, 60))
cv2.rectangle(img, (220, 180), (380, 280), (60, 60, 60), -1)
pts2 = np.array([[450, 170], [560, 200], [540, 300], [430, 280]], np.int32)
cv2.fillPoly(img, [pts2], (60, 60, 60))
cv2.circle(img, (100, 340), 40, (60, 60, 60), -1)
cv2.ellipse(img, (300, 340), (80, 30), -10, 0, 360, (60, 60, 60), -1)
cv2.rectangle(img, (450, 310), (570, 380), (60, 60, 60), -1)

# 检测
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 分析
stats = []
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / h if h > 0 else 0
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    stats.append({
        'id': i, 'area': area, 'perimeter': perimeter,
        'bbox': (x, y, w, h), 'aspect': aspect, 'circularity': circularity
    })

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('轮廓检测综合演示', fontsize=14, fontweight='bold')

# 原图
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

# 二值图
axes[0, 1].imshow(binary, cmap='gray')
axes[0, 1].set_title('二值图像')
axes[0, 1].axis('off')

# 轮廓+编号
canvas = img.copy()
colors = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 165, 0),
          (128, 0, 128), (0, 128, 128), (255, 0, 128), (0, 128, 255), (128, 128, 0)]
for i, cnt in enumerate(contours):
    cv2.drawContours(canvas, [cnt], 0, colors[i % len(colors)], 2)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(canvas, str(i), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

axes[0, 2].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f'检测到 {len(contours)} 个轮廓')
axes[0, 2].axis('off')

# 面积柱状图
areas = [s['area'] for s in stats]
axes[1, 0].bar(range(len(areas)), areas, color='steelblue', edgecolor='black')
axes[1, 0].set_xlabel('轮廓编号')
axes[1, 0].set_ylabel('面积')
axes[1, 0].set_title('轮廓面积')

# 圆度柱状图
circs = [s['circularity'] for s in stats]
axes[1, 1].bar(range(len(circs)), circs, color='coral', edgecolor='black')
axes[1, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('轮廓编号')
axes[1, 1].set_ylabel('圆度')
axes[1, 1].set_title('圆度 (>0.8 为近圆)')

# 统计汇总
summary = f"轮廓检测汇总\n{'='*35}\n"
summary += f"总轮廓数: {len(contours)}\n"
summary += f"总面积: {sum(areas):.0f}\n"
summary += f"平均面积: {np.mean(areas):.0f}\n"
summary += f"最大面积: {max(areas):.0f}\n"
summary += f"最小面积: {min(areas):.0f}\n\n"
summary += f"{'ID':>3} {'面积':>8} {'圆度':>6}\n"
summary += "-" * 20 + "\n"
for s in stats:
    summary += f"{s['id']:>3} {s['area']:>8.0f} {s['circularity']:>6.3f}\n"

axes[1, 2].text(0.05, 0.95, summary, fontsize=9, family='monospace',
                verticalalignment='top', transform=axes[1, 2].transAxes)
axes[1, 2].axis('off')
axes[1, 2].set_title('统计汇总')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_contour_demo.png'), dpi=150, bbox_inches='tight')
plt.show()
