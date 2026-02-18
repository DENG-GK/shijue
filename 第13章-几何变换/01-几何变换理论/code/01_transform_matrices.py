"""
示例1：理解变换矩阵
- 齐次坐标表示
- 平移/旋转/缩放/错切/组合变换矩阵
- 可视化变换前后的图形
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_shape():
    """创建带角标的正方形顶点"""
    points = np.array([
        [50, 50], [150, 50], [150, 150], [50, 150], [50, 50],
        [75, 50], [50, 75]
    ], dtype=np.float32)
    return points


original = create_shape()

# 定义各类变换矩阵
transformations = {
    '原始': np.eye(3, dtype=np.float32),

    '平移 (+50, +30)': np.array([
        [1, 0, 50],
        [0, 1, 30],
        [0, 0, 1]
    ], dtype=np.float32),

    '旋转 (30°)': np.array([
        [np.cos(np.radians(30)), -np.sin(np.radians(30)), 0],
        [np.sin(np.radians(30)), np.cos(np.radians(30)), 0],
        [0, 0, 1]
    ], dtype=np.float32),

    '缩放 (1.5x, 0.8x)': np.array([
        [1.5, 0, 0],
        [0, 0.8, 0],
        [0, 0, 1]
    ], dtype=np.float32),

    '错切 X': np.array([
        [1, 0.3, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32),

    '组合变换': np.array([
        [1.2 * np.cos(np.radians(15)), -1.2 * np.sin(np.radians(15)), 30],
        [1.2 * np.sin(np.radians(15)), 1.2 * np.cos(np.radians(15)), 20],
        [0, 0, 1]
    ], dtype=np.float32),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('几何变换矩阵可视化', fontsize=14, fontweight='bold')

for i, (name, matrix) in enumerate(transformations.items()):
    row, col = i // 3, i % 3
    ax = axes[row, col]

    # 齐次坐标变换
    ones = np.ones((original.shape[0], 1))
    points_h = np.hstack([original, ones])
    transformed_h = points_h @ matrix.T
    transformed = transformed_h[:, :2]

    # 绘制原始和变换后的图形
    ax.plot(original[:5, 0], original[:5, 1], 'b-', linewidth=2, label='原始')
    ax.plot(original[5:, 0], original[5:, 1], 'b-', linewidth=2)
    ax.fill(original[:4, 0], original[:4, 1], 'blue', alpha=0.15)

    ax.plot(transformed[:5, 0], transformed[:5, 1], 'r-', linewidth=2, label='变换后')
    ax.plot(transformed[5:, 0], transformed[5:, 1], 'r-', linewidth=2)
    ax.fill(transformed[:4, 0], transformed[:4, 1], 'red', alpha=0.15)

    ax.set_xlim(-50, 300)
    ax.set_ylim(-50, 250)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{name}\n{matrix[:2, :].round(2)}', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_transform_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()

print("变换矩阵可视化完成！")
print("变换类型: 平移、旋转、缩放、错切、组合")
