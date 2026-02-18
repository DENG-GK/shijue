"""
示例8：融合质量分析
- 不同pyramid levels(1-7)的效果
- 边缘不连续性度量
- 最佳层数分析
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def build_gaussian_pyramid(image, levels):
    pyramid = [image.astype(np.float64)]
    current = image.astype(np.float64)
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid(image, levels):
    G = build_gaussian_pyramid(image, levels)
    L = []
    for i in range(levels - 1):
        expanded = cv2.pyrUp(G[i + 1])
        if expanded.shape != G[i].shape:
            expanded = cv2.resize(expanded, (G[i].shape[1], G[i].shape[0]))
        L.append(G[i] - expanded)
    L.append(G[-1])
    return L


def reconstruct_from_laplacian(pyramid):
    result = pyramid[-1].copy()
    for i in range(len(pyramid) - 2, -1, -1):
        expanded = cv2.pyrUp(result)
        if expanded.shape != pyramid[i].shape:
            expanded = cv2.resize(expanded, (pyramid[i].shape[1], pyramid[i].shape[0]))
        result = expanded + pyramid[i]
    return result


def pyramid_blend(img1, img2, mask, levels=6):
    L1 = build_laplacian_pyramid(img1, levels)
    L2 = build_laplacian_pyramid(img2, levels)
    GM = build_gaussian_pyramid(mask, levels)
    L_blend = []
    for l1, l2, gm in zip(L1, L2, GM):
        if len(l1.shape) == 3 and len(gm.shape) == 2:
            gm = np.stack([gm] * 3, axis=2)
        L_blend.append(l1 * gm + l2 * (1 - gm))
    return np.clip(reconstruct_from_laplacian(L_blend), 0, 255).astype(np.uint8)


size = (256, 256)
img1 = np.zeros((*size, 3), dtype=np.uint8)
img1[:, :] = [0, 0, 200]
cv2.rectangle(img1, (50, 50), (150, 150), (0, 0, 150), -1)

img2 = np.zeros((*size, 3), dtype=np.uint8)
img2[:, :] = [200, 0, 0]
cv2.circle(img2, (128, 128), 60, (150, 0, 0), -1)

mask = np.zeros(size, dtype=np.float64)
mask[:, :size[1] // 2] = 1.0

levels_range = range(1, 8)
results = []
for levels in levels_range:
    blended = pyramid_blend(img1, img2, mask, levels)
    gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    mask_edge = cv2.Canny((mask * 255).astype(np.uint8), 100, 200)
    edge_region = mask_edge > 0
    edge_disc = np.mean(np.abs(lap[edge_region])) if np.sum(edge_region) > 0 else 0
    smoothness = np.std(lap)

    results.append({'levels': levels, 'blended': blended,
                    'edge_disc': edge_disc, 'smoothness': smoothness})

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('融合质量 vs 金字塔层数', fontsize=14, fontweight='bold')

for i, r in enumerate(results[:7]):
    row, col = i // 4, i % 4
    axes[row, col].imshow(cv2.cvtColor(r['blended'], cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f"Levels={r['levels']}")
    axes[row, col].axis('off')

# 指标图
axes[1, 3].plot([r['levels'] for r in results], [r['edge_disc'] for r in results],
                'ro-', linewidth=2, label='接缝可见度')
axes[1, 3].set_xlabel('金字塔层数')
axes[1, 3].set_ylabel('边缘不连续性')
axes[1, 3].set_title('质量分析\n（越低越好）')
axes[1, 3].legend(fontsize=8)
axes[1, 3].grid(True, alpha=0.3)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_blend_quality.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n融合质量分析:")
print(f"{'层数':<6} {'接缝可见度':<15} {'平滑度':<12}")
for r in results:
    print(f"{r['levels']:<6} {r['edge_disc']:<15.4f} {r['smoothness']:<12.4f}")
