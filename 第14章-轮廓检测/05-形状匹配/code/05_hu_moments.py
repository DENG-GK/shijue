"""
示例5：Hu矩详解
- 计算各形状的Hu矩
- 对数变换便于比较
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建不同形状
shapes = {}
# 圆
img = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(img, (100, 100), 60, 255, -1)
shapes["Circle"] = img
# 正方形
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (40, 40), (160, 160), 255, -1)
shapes["Square"] = img
# 矩形
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (30, 60), (170, 140), 255, -1)
shapes["Rectangle"] = img
# 三角形
img = np.zeros((200, 200), dtype=np.uint8)
cv2.fillPoly(img, [np.array([[100, 30], [170, 170], [30, 170]])], 255)
shapes["Triangle"] = img
# 椭圆
img = np.zeros((200, 200), dtype=np.uint8)
cv2.ellipse(img, (100, 100), (80, 40), 0, 0, 360, 255, -1)
shapes["Ellipse"] = img

# 分析
print("Hu矩分析 (对数变换后):")
print("=" * 90)
header = f"{'形状':>12}"
for i in range(7):
    header += f"   Hu{i + 1:>3}"
print(header)
print("-" * 90)

hu_data = {}
for name, simg in shapes.items():
    cnts, _ = cv2.findContours(simg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue
    moments = cv2.moments(cnts[0])
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    hu_data[name] = hu_log
    row = f"{name:>12}"
    for h in hu_log:
        row += f"  {h:>7.2f}"
    print(row)

print("\nHu矩特性说明:")
print("  Hu1: 分散程度  |  Hu2: 对称性  |  Hu3-4: 扭曲  |  Hu5-7: 非对称性")

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Hu矩详解', fontsize=14, fontweight='bold')

# 上：形状预览
ax0 = axes[0]
combined = np.zeros((200, 200 * len(shapes), 3), dtype=np.uint8)
for idx, (name, simg) in enumerate(shapes.items()):
    rgb = cv2.cvtColor(simg, cv2.COLOR_GRAY2BGR)
    combined[:, idx * 200:(idx + 1) * 200] = rgb
ax0.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
# 标注名称
for idx, name in enumerate(shapes.keys()):
    ax0.text(idx * 200 + 100, 15, name, ha='center', va='top',
             fontsize=11, color='lime', fontweight='bold')
ax0.set_title('形状预览')
ax0.axis('off')

# 下：Hu矩柱状图
ax1 = axes[1]
x = np.arange(7)
width = 0.15
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for idx, (name, hu_log) in enumerate(hu_data.items()):
    ax1.bar(x + idx * width, hu_log, width, label=name, color=colors[idx % len(colors)])
ax1.set_xlabel('Hu矩编号')
ax1.set_ylabel('对数变换值')
ax1.set_title('各形状Hu矩对比 (对数)')
ax1.set_xticks(x + width * 2)
ax1.set_xticklabels([f'Hu{i + 1}' for i in range(7)])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_hu_moments.png'), dpi=150, bbox_inches='tight')
plt.show()
