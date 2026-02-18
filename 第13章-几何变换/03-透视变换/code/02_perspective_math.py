"""
示例2：透视变换的数学原理
- 3x3单应矩阵 H
- 齐次坐标归一化 x'=x_h/w', y'=y_h/w'
- 网格变换可视化
- 逆变换恢复
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建网格图
grid_size = 200
grid = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 255
for i in range(0, grid_size, 20):
    cv2.line(grid, (i, 0), (i, grid_size - 1), (200, 200, 200), 1)
    cv2.line(grid, (0, i), (grid_size - 1, i), (200, 200, 200), 1)
cv2.line(grid, (0, 0), (grid_size - 1, grid_size - 1), (0, 0, 255), 1)
cv2.circle(grid, (grid_size // 2, grid_size // 2), 5, (255, 0, 0), -1)

src_pts = np.float32([[0, 0], [grid_size - 1, 0],
                       [grid_size - 1, grid_size - 1], [0, grid_size - 1]])
dst_pts = np.float32([[30, 30], [170, 20], [190, 180], [10, 170]])

M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(grid, M, (grid_size, grid_size))

# 逆变换
M_inv = np.linalg.inv(M)
recovered = cv2.warpPerspective(warped, M_inv, (grid_size, grid_size))
error = cv2.absdiff(grid, recovered)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('透视变换数学原理', fontsize=14, fontweight='bold')

# 原始网格
axes[0, 0].imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
for i, pt in enumerate(src_pts):
    axes[0, 0].plot(pt[0], pt[1], 'go', markersize=10)
    axes[0, 0].annotate(f'P{i + 1}', (pt[0] + 5, pt[1] + 5), fontsize=10)
axes[0, 0].set_title('原始网格')
axes[0, 0].axis('off')

# 变换后
axes[0, 1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
for i, pt in enumerate(dst_pts):
    axes[0, 1].plot(pt[0], pt[1], 'ro', markersize=10)
    axes[0, 1].annotate(f"P{i + 1}'", (pt[0] + 5, pt[1] + 5), fontsize=10)
axes[0, 1].set_title('透视变换后')
axes[0, 1].axis('off')

# 变换矩阵
text = "变换矩阵 H:\n\n"
for i in range(3):
    text += f"[{M[i, 0]:8.4f}  {M[i, 1]:8.4f}  {M[i, 2]:8.4f}]\n"
text += "\n齐次坐标归一化:\n"
text += "x_final = x'/w'\n"
text += "y_final = y'/w'"
axes[0, 2].text(0.1, 0.5, text, fontsize=10, family='monospace',
                verticalalignment='center', transform=axes[0, 2].transAxes)
axes[0, 2].axis('off')
axes[0, 2].set_title('变换矩阵 H')

# 点映射
test_points = np.float32([[50, 50], [150, 50], [100, 100], [50, 150], [150, 150]])
axes[1, 0].set_title('点映射可视化')
for pt in test_points:
    pt_h = np.array([pt[0], pt[1], 1])
    transformed_h = M @ pt_h
    t = transformed_h[:2] / transformed_h[2]
    axes[1, 0].plot([pt[0], t[0]], [pt[1], t[1]], 'b-', alpha=0.5)
    axes[1, 0].plot(pt[0], pt[1], 'go', markersize=8)
    axes[1, 0].plot(t[0], t[1], 'ro', markersize=8)
axes[1, 0].set_xlim(-10, 210)
axes[1, 0].set_ylim(210, -10)
axes[1, 0].legend(['映射', '原始', '变换后'], loc='upper right', fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# 逆变换
axes[1, 1].imshow(cv2.cvtColor(recovered, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('逆变换恢复')
axes[1, 1].axis('off')

# 误差
axes[1, 2].imshow(cv2.cvtColor(error * 10, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f'误差 (10x)\n最大: {error.max()}')
axes[1, 2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_perspective_math.png'), dpi=150, bbox_inches='tight')
plt.show()

print("透视变换数学原理演示完成！")
