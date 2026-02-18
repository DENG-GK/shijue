"""
示例2：拉普拉斯金字塔可视化增强
- 第1行：高斯金字塔
- 第2行：拉普拉斯金字塔（归一化显示）
- 第3行：绝对值边缘图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

levels = 4

# 创建测试图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.rectangle(image, (20, 20), (236, 236), (0, 100, 200), -1)
cv2.circle(image, (128, 128), 70, (200, 50, 50), -1)
cv2.rectangle(image, (80, 80), (176, 176), (50, 200, 50), 3)
for i in range(32):
    for j in range(32):
        if (i + j) % 2 == 0:
            cv2.rectangle(image, (i * 8, j * 8), (i * 8 + 4, j * 8 + 4), (150, 150, 150), -1)

# 构建高斯金字塔
G = [image.astype(np.float64)]
for i in range(levels):
    G.append(cv2.pyrDown(G[-1]))

# 构建拉普拉斯金字塔
L = []
for i in range(levels):
    expanded = cv2.pyrUp(G[i + 1])
    if expanded.shape != G[i].shape:
        expanded = cv2.resize(expanded, (G[i].shape[1], G[i].shape[0]))
    L.append(G[i] - expanded)
L.append(G[-1])

fig, axes = plt.subplots(3, levels + 1, figsize=(18, 12))
fig.suptitle('拉普拉斯金字塔可视化', fontsize=14, fontweight='bold')

# 高斯金字塔
for i, g in enumerate(G):
    display = np.clip(g, 0, 255).astype(np.uint8)
    axes[0, i].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'G{i}: {g.shape[1]}×{g.shape[0]}', fontsize=9)
    axes[0, i].axis('off')

# 拉普拉斯金字塔（归一化）
for i, lap in enumerate(L):
    if i < levels:
        lap_norm = lap - lap.min()
        if lap_norm.max() > 0:
            lap_norm = lap_norm / lap_norm.max() * 255
        display = lap_norm.astype(np.uint8)
    else:
        display = np.clip(lap, 0, 255).astype(np.uint8)
    axes[1, i].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f'L{i} (归一化)', fontsize=9)
    axes[1, i].axis('off')

# 绝对值（边缘）
for i, lap in enumerate(L[:-1]):
    lap_abs = np.abs(lap)
    lap_enhanced = np.clip(lap_abs * 2, 0, 255).astype(np.uint8)
    axes[2, i].imshow(cv2.cvtColor(lap_enhanced, cv2.COLOR_BGR2RGB))
    axes[2, i].set_title(f'|L{i}| (边缘)', fontsize=9)
    axes[2, i].axis('off')

# 最后一个子图说明
axes[2, levels].text(0.1, 0.5,
                     "拉普拉斯金字塔:\n\n行1: 高斯金字塔\n（逐级模糊+下采样）\n\n"
                     "行2: 拉普拉斯层\n（相邻高斯层差值）\n\n"
                     "行3: 绝对值\n（突出边缘和纹理）",
                     fontsize=9, va='center', transform=axes[2, levels].transAxes)
axes[2, levels].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_laplacian_visualization.png'), dpi=150, bbox_inches='tight')
plt.show()
