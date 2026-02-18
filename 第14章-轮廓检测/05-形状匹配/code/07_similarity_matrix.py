"""
示例7：形状相似度矩阵
- 多形状间相互相似度计算
- 热力图可视化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_shapes():
    shapes = {}
    # 圆
    img = np.zeros((100, 100), np.uint8)
    cv2.circle(img, (50, 50), 40, 255, -1)
    shapes["Circle"] = img
    # 椭圆
    img = np.zeros((100, 100), np.uint8)
    cv2.ellipse(img, (50, 50), (45, 30), 0, 0, 360, 255, -1)
    shapes["Ellipse"] = img
    # 正方形
    img = np.zeros((100, 100), np.uint8)
    cv2.rectangle(img, (15, 15), (85, 85), 255, -1)
    shapes["Square"] = img
    # 矩形
    img = np.zeros((100, 100), np.uint8)
    cv2.rectangle(img, (10, 25), (90, 75), 255, -1)
    shapes["Rect"] = img
    # 三角形
    img = np.zeros((100, 100), np.uint8)
    cv2.fillPoly(img, [np.array([[50, 10], [90, 90], [10, 90]])], 255)
    shapes["Triangle"] = img
    # 五边形
    img = np.zeros((100, 100), np.uint8)
    pts = []
    for i in range(5):
        a = -np.pi / 2 + i * 2 * np.pi / 5
        pts.append([int(50 + 40 * np.cos(a)), int(50 + 40 * np.sin(a))])
    cv2.fillPoly(img, [np.array(pts)], 255)
    shapes["Pentagon"] = img
    # 六边形
    img = np.zeros((100, 100), np.uint8)
    pts = []
    for i in range(6):
        a = i * np.pi / 3
        pts.append([int(50 + 40 * np.cos(a)), int(50 + 40 * np.sin(a))])
    cv2.fillPoly(img, [np.array(pts)], 255)
    shapes["Hexagon"] = img
    # 星形
    img = np.zeros((100, 100), np.uint8)
    pts = []
    for i in range(5):
        a_o = -np.pi / 2 + i * 2 * np.pi / 5
        a_i = -np.pi / 2 + (i + 0.5) * 2 * np.pi / 5
        pts.append([int(50 + 40 * np.cos(a_o)), int(50 + 40 * np.sin(a_o))])
        pts.append([int(50 + 18 * np.cos(a_i)), int(50 + 18 * np.sin(a_i))])
    cv2.fillPoly(img, [np.array(pts)], 255)
    shapes["Star"] = img
    return shapes


shapes = create_shapes()
contours = {}
for name, img in shapes.items():
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours[name] = cnts[0]

names = list(shapes.keys())
n = len(names)

# 计算相似度矩阵
sim = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim[i, j] = cv2.matchShapes(contours[names[i]], contours[names[j]],
                                     cv2.CONTOURS_MATCH_I1, 0)

print("形状相似度矩阵 (I1, 值越小越相似):")
print("=" * 90)
header = f"{'':>10}"
for name in names:
    header += f"{name:>10}"
print(header)
print("-" * 90)
for i, name in enumerate(names):
    row = f"{name:>10}"
    for j in range(n):
        row += f"{sim[i, j]:>10.4f}"
    print(row)

# 最相似的对
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((names[i], names[j], sim[i, j]))
pairs.sort(key=lambda x: x[2])
print("\n最相似的形状对:")
for n1, n2, s in pairs[:5]:
    print(f"  {n1} <-> {n2}: {s:.4f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('形状相似度矩阵', fontsize=14, fontweight='bold')

# 形状预览
combined = np.zeros((100, 100 * len(shapes), 3), dtype=np.uint8)
for idx, (name, img) in enumerate(shapes.items()):
    combined[:, idx * 100:(idx + 1) * 100] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
axes[0].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
for idx, name in enumerate(shapes.keys()):
    axes[0].text(idx * 100 + 50, 8, name, ha='center', fontsize=8, color='lime')
axes[0].set_title('形状预览')
axes[0].axis('off')

# 热力图
im = axes[1].imshow(sim, cmap='YlOrRd', aspect='auto')
axes[1].set_xticks(range(n))
axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
axes[1].set_yticks(range(n))
axes[1].set_yticklabels(names, fontsize=9)
axes[1].set_title('相似度热力图 (值越小越相似)')
# 标注数值
for i in range(n):
    for j in range(n):
        axes[1].text(j, i, f'{sim[i, j]:.3f}', ha='center', va='center', fontsize=7,
                     color='white' if sim[i, j] > sim.max() * 0.5 else 'black')
plt.colorbar(im, ax=axes[1], shrink=0.8)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_similarity_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
