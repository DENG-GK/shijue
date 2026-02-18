"""
示例2：图像矩
- 空间矩、中心矩、Hu矩
- 质心计算
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

shapes = {}
img_sq = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img_sq, (50, 50), (150, 150), 255, -1)
shapes["正方形"] = img_sq

img_ci = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(img_ci, (100, 100), 50, 255, -1)
shapes["圆形"] = img_ci

img_tr = np.zeros((200, 200), dtype=np.uint8)
cv2.fillPoly(img_tr, [np.array([[100, 30], [170, 170], [30, 170]])], 255)
shapes["三角形"] = img_tr

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('图像矩与Hu矩', fontsize=14, fontweight='bold')

hu_data = {}
for idx, (name, img) in enumerate(shapes.items()):
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        M = cv2.moments(cnts[0])
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        hu = cv2.HuMoments(M).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        hu_data[name] = hu_log

        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(canvas, (cx, cy), 5, (0, 0, 255), -1)
        cv2.drawContours(canvas, cnts, 0, (0, 255, 0), 2)
        axes[0, idx].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        axes[0, idx].set_title(f'{name}\n质心: ({cx},{cy})')
        axes[0, idx].axis('off')

# Hu矩对比柱状图
x = np.arange(7)
width = 0.25
for i, (name, hu_log) in enumerate(hu_data.items()):
    axes[1, 0].bar(x + i * width, hu_log, width, label=name, alpha=0.8)
axes[1, 0].set_xticks(x + width)
axes[1, 0].set_xticklabels([f'Hu{i+1}' for i in range(7)])
axes[1, 0].set_ylabel('对数值')
axes[1, 0].set_title('Hu矩对比 (对数)')
axes[1, 0].legend(fontsize=8)

# 矩信息文本
info = "Hu矩对数值:\n" + "=" * 50 + "\n"
info += f"{'形状':>6}"
for i in range(7):
    info += f" Hu{i+1:>6}"
info += "\n" + "-" * 50 + "\n"
for name, hu_log in hu_data.items():
    info += f"{name:>6}"
    for h in hu_log:
        info += f" {h:>6.2f}"
    info += "\n"

axes[1, 1].text(0.05, 0.95, info, fontsize=8, family='monospace',
                verticalalignment='top', transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')
axes[1, 1].set_title('Hu矩数值')

axes[1, 2].text(0.05, 0.5,
    "Hu矩特性:\n\n"
    "- 平移不变性\n"
    "- 缩放不变性\n"
    "- 旋转不变性\n\n"
    "Hu1: 形状分散程度\n"
    "Hu2: 对称性\n"
    "Hu3-4: 扭曲程度\n"
    "Hu5-7: 非对称性",
    fontsize=10, verticalalignment='center', transform=axes[1, 2].transAxes)
axes[1, 2].axis('off')
axes[1, 2].set_title('特性说明')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_moments.png'), dpi=150, bbox_inches='tight')
plt.show()
