"""
示例9：交互式轮廓近似
- matplotlib Slider 调整 epsilon
- 实时预览近似效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建花形轮廓
img = np.zeros((400, 500), dtype=np.uint8)
pts = []
for i in range(36):
    angle = i * 10 * np.pi / 180
    r = 100 + 30 * np.sin(5 * angle)
    x = int(250 + r * np.cos(angle))
    y = int(200 + r * np.sin(angle))
    pts.append([x, y])
cv2.fillPoly(img, [np.array(pts)], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
perimeter = cv2.arcLength(cnt, True)

# 预计算多个 epsilon 的结果用于静态展示
ratios = [0.5, 1, 2, 5, 8, 15]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('交互式轮廓近似 - 不同epsilon效果', fontsize=14, fontweight='bold')

for idx, pct in enumerate(ratios):
    epsilon = pct / 100.0 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    canvas = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(canvas, [cnt], 0, (60, 60, 60), 1)
    cv2.drawContours(canvas, [approx], 0, (0, 255, 0), 2)
    for pt in approx:
        cv2.circle(canvas, tuple(pt[0]), 4, (0, 0, 255), -1)

    row, col = idx // 3, idx % 3
    axes[row, col].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    compress = len(approx) / len(cnt) * 100
    axes[row, col].set_title(
        f'ε={pct}% → {len(approx)}点 (压缩{compress:.1f}%)')
    axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_interactive_approx.png'), dpi=150, bbox_inches='tight')

# 交互式 Slider 界面
fig2, ax2 = plt.subplots(figsize=(8, 7))
plt.subplots_adjust(bottom=0.2)
ax2.set_title('拖动滑块调整 epsilon')
ax2.axis('off')

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.04])
slider = Slider(ax_slider, 'ε (%)', 0.1, 20.0, valinit=2.0, valstep=0.1)


def update(val):
    eps = val / 100.0 * perimeter
    approx = cv2.approxPolyDP(cnt, eps, True)

    canvas = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(canvas, [cnt], 0, (60, 60, 60), 1)
    cv2.drawContours(canvas, [approx], 0, (0, 255, 0), 2)
    for pt in approx:
        cv2.circle(canvas, tuple(pt[0]), 4, (0, 0, 255), -1)

    ax2.clear()
    ax2.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'ε={val:.1f}% → {len(approx)}点 (原{len(cnt)}点, 压缩{len(approx)/len(cnt)*100:.1f}%)')
    ax2.axis('off')
    fig2.canvas.draw_idle()


slider.on_changed(update)
update(2.0)
plt.show()
