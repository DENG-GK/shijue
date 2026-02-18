"""
示例6：拉普拉斯金字塔图像压缩概念
- 阈值处理细节层（小值置零）
- 不同threshold效果对比
- PSNR vs threshold 分析
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def build_laplacian_pyramid(image, levels=4):
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


def compress_pyramid(image, levels=4, threshold=10):
    """压缩：将拉普拉斯金字塔中小值置零"""
    lap_pyr = build_laplacian_pyramid(image, levels)
    compressed = []
    sparsity = []
    for i, level in enumerate(lap_pyr):
        if i < len(lap_pyr) - 1:
            comp = level.copy()
            comp[np.abs(comp) < threshold] = 0
            zero_ratio = np.sum(np.abs(comp) == 0) / comp.size
        else:
            comp = level
            zero_ratio = 0
        compressed.append(comp)
        sparsity.append(zero_ratio * 100)
    recon = reconstruct_from_laplacian(compressed)
    return np.clip(recon, 0, 255).astype(np.uint8), sparsity


# 创建测试图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.rectangle(image, (30, 30), (226, 226), (100, 150, 200), -1)
cv2.circle(image, (128, 128), 60, (200, 100, 50), -1)
noise = np.random.randint(0, 20, image.shape, dtype=np.uint8)
image = cv2.add(image, noise)

thresholds = [0, 5, 10, 20, 40]
results = []
for thresh in thresholds:
    recon, sparsity = compress_pyramid(image, threshold=thresh)
    error = cv2.absdiff(image, recon)
    psnr = cv2.PSNR(image, recon) if np.max(error) > 0 else float('inf')
    results.append((thresh, recon, psnr, sparsity))

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle('拉普拉斯金字塔压缩', fontsize=14, fontweight='bold')

for i, (thresh, recon, psnr, _) in enumerate(results):
    axes[0, i].imshow(cv2.cvtColor(recon, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'阈值={thresh}\nPSNR: {psnr:.1f} dB')
    axes[0, i].axis('off')

    error = cv2.absdiff(image, recon)
    error_display = cv2.convertScaleAbs(error, alpha=5)
    axes[1, i].imshow(cv2.cvtColor(error_display, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title('误差(5x)')
    axes[1, i].axis('off')

# PSNR曲线
psnrs = [r[2] for r in results]
axes[2, 0].plot(thresholds, psnrs, 'go-', linewidth=2, markersize=8)
axes[2, 0].set_xlabel('阈值')
axes[2, 0].set_ylabel('PSNR (dB)')
axes[2, 0].set_title('质量 vs 阈值')
axes[2, 0].grid(True, alpha=0.3)

# 各层稀疏度
for idx, (thresh, _, _, sparsity) in enumerate(results[1:], 1):
    axes[2, 1].plot(range(len(sparsity) - 1), sparsity[:-1], 'o-', label=f'T={thresh}')
axes[2, 1].set_xlabel('金字塔层级')
axes[2, 1].set_ylabel('零值比例 (%)')
axes[2, 1].set_title('稀疏度')
axes[2, 1].legend(fontsize=8)
axes[2, 1].grid(True, alpha=0.3)

for j in range(2, 5):
    axes[2, j].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_compression_demo.png'), dpi=150, bbox_inches='tight')
plt.show()
