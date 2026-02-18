"""
示例6：变换组合
- 平移/旋转/缩放的矩阵乘法组合
- 验证变换顺序的影响(非交换性)
- T_combined = T_n @ T_{n-1} @ ... @ T_1
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建L形
points = np.array([
    [100, 100], [200, 100], [200, 150], [150, 150],
    [150, 200], [100, 200], [100, 100]
], dtype=np.float32)


def apply_transform(pts, matrix):
    """对点集应用变换矩阵"""
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    transformed = pts_h @ matrix.T
    return transformed[:, :2]


# 定义基本变换
T_translate = np.array([[1, 0, 50], [0, 1, 30], [0, 0, 1]], dtype=np.float32)

angle = np.radians(30)
T_rotate = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
], dtype=np.float32)

T_scale = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]], dtype=np.float32)

# 不同组合顺序
T_order1 = T_scale @ T_rotate @ T_translate  # 先平移→旋转→缩放
T_order2 = T_translate @ T_rotate @ T_scale  # 先缩放→旋转→平移
T_order3 = T_scale @ T_translate @ T_rotate  # 先旋转→平移→缩放

results = {
    '原始': points,
    '仅平移': apply_transform(points, T_translate),
    '仅旋转': apply_transform(points, T_rotate),
    '仅缩放': apply_transform(points, T_scale),
    '平移→旋转→缩放': apply_transform(points, T_order1),
    '缩放→旋转→平移': apply_transform(points, T_order2),
    '旋转→平移→缩放': apply_transform(points, T_order3),
}

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('变换组合与顺序影响', fontsize=14, fontweight='bold')

colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

for i, (name, pts) in enumerate(results.items()):
    row, col = i // 4, i % 4
    ax = axes[row, col]

    # 原始参考
    ax.plot(points[:, 0], points[:, 1], 'k--', linewidth=1, alpha=0.3)
    ax.fill(points[:-1, 0], points[:-1, 1], 'gray', alpha=0.1)

    # 变换后
    ax.plot(pts[:, 0], pts[:, 1], '-', linewidth=2, color=colors[i])
    ax.fill(pts[:-1, 0], pts[:-1, 1], color=colors[i], alpha=0.3)

    ax.set_xlim(-50, 400)
    ax.set_ylim(-50, 400)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(name)

# 对比图
ax = axes[1, 3]
ax.plot(results['平移→旋转→缩放'][:, 0], results['平移→旋转→缩放'][:, 1],
        'r-', linewidth=2, label='平移→旋转→缩放')
ax.plot(results['缩放→旋转→平移'][:, 0], results['缩放→旋转→平移'][:, 1],
        'b-', linewidth=2, label='缩放→旋转→平移')
ax.plot(results['旋转→平移→缩放'][:, 0], results['旋转→平移→缩放'][:, 1],
        'g-', linewidth=2, label='旋转→平移→缩放')
ax.set_xlim(-50, 400)
ax.set_ylim(-50, 400)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.set_title('顺序影响对比!')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_transform_composition.png'), dpi=150, bbox_inches='tight')
plt.show()

print("变换组合原理:")
print("  T_combined = T_n @ T_{n-1} @ ... @ T_1")
print("  变换从右到左应用！")
print("  注意：矩阵乘法不满足交换律，顺序很重要！")
