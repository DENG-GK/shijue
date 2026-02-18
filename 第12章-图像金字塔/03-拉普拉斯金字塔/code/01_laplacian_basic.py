"""
示例1：基本拉普拉斯金字塔构建
- L_i = G_i - expand(G_{i+1})
- 每层包含特定频带的细节信息
- 统计每层的均值和标准差
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def build_laplacian_pyramid(image, levels=5):
    """构建拉普拉斯金字塔"""
    G = [image.astype(np.float64)]
    current = image.astype(np.float64)
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        G.append(current)

    L = []
    for i in range(levels - 1):
        expanded = cv2.pyrUp(G[i + 1])
        if expanded.shape != G[i].shape:
            expanded = cv2.resize(expanded, (G[i].shape[1], G[i].shape[0]))
        L.append(G[i] - expanded)
    L.append(G[-1])  # 顶层保留低频
    return L


# 创建测试图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.rectangle(image, (30, 30), (220, 220), (100, 150, 200), -1)
cv2.circle(image, (128, 128), 60, (200, 100, 50), -1)
for i in range(20):
    cv2.line(image, (50 + i * 10, 180), (50 + i * 10, 230), (255, 255, 255), 1)

lap_pyr = build_laplacian_pyramid(image, levels=5)

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('拉普拉斯金字塔构建', fontsize=14, fontweight='bold')

for i, level in enumerate(lap_pyr):
    if i < len(lap_pyr) - 1:
        display = level + 128
        display = np.clip(display, 0, 255).astype(np.uint8)
    else:
        display = np.clip(level, 0, 255).astype(np.uint8)

    axes[0, i].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'L{i}: {level.shape[1]}×{level.shape[0]}')
    axes[0, i].axis('off')

# 统计信息
means = [np.mean(l) for l in lap_pyr]
stds = [np.std(l) for l in lap_pyr]

axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('原始图像')
axes[1, 0].axis('off')

axes[1, 1].bar(range(len(means)), means, color='steelblue', alpha=0.7)
axes[1, 1].set_xlabel('层级')
axes[1, 1].set_ylabel('均值')
axes[1, 1].set_title('各层均值（接近0）')

axes[1, 2].bar(range(len(stds)), stds, color='coral', alpha=0.7)
axes[1, 2].set_xlabel('层级')
axes[1, 2].set_ylabel('标准差')
axes[1, 2].set_title('各层标准差')

# 能量分布
energies = [np.sum(l ** 2) for l in lap_pyr]
axes[1, 3].pie(energies, labels=[f'L{i}' for i in range(len(energies))],
               autopct='%1.1f%%', startangle=90)
axes[1, 3].set_title('能量分布')

axes[1, 4].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_laplacian_basic.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n拉普拉斯金字塔统计:")
for i, level in enumerate(lap_pyr):
    print(f"L{i}: Shape={level.shape[:2]}, Mean={np.mean(level):.2f}, "
          f"Std={np.std(level):.2f}, Range=[{np.min(level):.1f}, {np.max(level):.1f}]")
