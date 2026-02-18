"""
示例3：完美重建
- 从拉普拉斯金字塔重建原始图像
- G_i = L_i + expand(G_{i+1})
- 误差分析、PSNR计算
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def build_laplacian_pyramid(image, levels=5):
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


# 创建测试图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.rectangle(image, (30, 30), (226, 226), (0, 150, 255), -1)
cv2.circle(image, (128, 128), 60, (255, 50, 50), -1)
cv2.putText(image, 'TEST', (70, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

levels = 5
lap_pyr = build_laplacian_pyramid(image, levels)
reconstructed = reconstruct_from_laplacian(lap_pyr)
reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

error = cv2.absdiff(image, reconstructed)
max_error = np.max(error)
mean_error = np.mean(error)
psnr = cv2.PSNR(image, reconstructed)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('拉普拉斯金字塔完美重建', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原图')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('重建图像')
axes[0, 1].axis('off')

error_display = cv2.convertScaleAbs(error, alpha=50)
axes[0, 2].imshow(cv2.cvtColor(error_display, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f'误差(50x放大)\nMax: {max_error}')
axes[0, 2].axis('off')

axes[0, 3].hist(error.flatten(), bins=50, color='steelblue', alpha=0.7)
axes[0, 3].set_xlabel('误差值')
axes[0, 3].set_ylabel('频率')
axes[0, 3].set_title(f'误差分布\nMean: {mean_error:.4f}')

# 重建过程
steps = [lap_pyr[-1]]
current = lap_pyr[-1].copy()
for i in range(len(lap_pyr) - 2, -1, -1):
    expanded = cv2.pyrUp(current)
    if expanded.shape != lap_pyr[i].shape:
        expanded = cv2.resize(expanded, (lap_pyr[i].shape[1], lap_pyr[i].shape[0]))
    current = expanded + lap_pyr[i]
    steps.append(current.copy())

for i, step in enumerate(steps[:4]):
    display = np.clip(step, 0, 255).astype(np.uint8)
    axes[1, i].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f'重建步骤 {i}')
    axes[1, i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_perfect_reconstruction.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n重建质量: Max Error={max_error}, Mean Error={mean_error:.6f}, PSNR={psnr:.2f} dB")
print(f"完美重建: {'是' if max_error <= 1 else '否'}")
