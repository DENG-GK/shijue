"""
示例10：金字塔分解与重组
- 完整的高斯-拉普拉斯分解
- 完美重建验证
- 误差直方图和PSNR
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
    image[:, i] = [i, 128, 255 - i]
cv2.circle(image, (128, 128), 60, (255, 255, 255), -1)
cv2.rectangle(image, (50, 50), (100, 100), (0, 0, 0), -1)

n_levels = 4

# 高斯金字塔
G = [image.astype(np.float64)]
for i in range(n_levels):
    G.append(cv2.pyrDown(G[-1]))

# 拉普拉斯金字塔
L = []
for i in range(n_levels):
    upsampled = cv2.pyrUp(G[i + 1])
    if upsampled.shape != G[i].shape:
        upsampled = cv2.resize(upsampled, (G[i].shape[1], G[i].shape[0]))
    L.append(G[i] - upsampled)
L.append(G[-1])

# 完美重建
reconstructed = L[-1].copy()
for i in range(n_levels - 1, -1, -1):
    upsampled = cv2.pyrUp(reconstructed)
    if upsampled.shape != L[i].shape:
        upsampled = cv2.resize(upsampled, (L[i].shape[1], L[i].shape[0]))
    reconstructed = upsampled + L[i]
reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

diff = cv2.absdiff(image, reconstructed)
psnr = cv2.PSNR(image, reconstructed)
mse = np.mean((image.astype(float) - reconstructed.astype(float)) ** 2)

fig = plt.figure(figsize=(18, 12))
fig.suptitle('金字塔分解与重组', fontsize=14, fontweight='bold')

# 高斯金字塔
for i, g in enumerate(G[:5]):
    ax = fig.add_subplot(3, 6, i + 1)
    ax.imshow(cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax.set_title(f'G{i}', fontsize=9)
    ax.axis('off')

# 拉普拉斯金字塔
for i, l in enumerate(L[:5]):
    ax = fig.add_subplot(3, 6, 7 + i)
    l_display = l - l.min()
    if l_display.max() > 0:
        l_display = (l_display / l_display.max() * 255).astype(np.uint8)
    else:
        l_display = l_display.astype(np.uint8)
    ax.imshow(cv2.cvtColor(l_display, cv2.COLOR_BGR2RGB))
    ax.set_title(f'L{i}', fontsize=9)
    ax.axis('off')

# 重建结果
ax = fig.add_subplot(3, 4, 9)
ax.imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
ax.set_title('重建')
ax.axis('off')

ax = fig.add_subplot(3, 4, 10)
diff_amp = cv2.convertScaleAbs(diff, alpha=10)
ax.imshow(cv2.cvtColor(diff_amp, cv2.COLOR_BGR2RGB))
ax.set_title(f'差异(10x)\nMax: {np.max(diff)}')
ax.axis('off')

ax = fig.add_subplot(3, 4, 11)
ax.hist(diff.flatten(), bins=50, color='steelblue', alpha=0.7)
ax.set_xlabel('误差值')
ax.set_title('误差分布')

ax = fig.add_subplot(3, 4, 12)
info = f"PSNR: {psnr:.2f} dB\nMSE: {mse:.4f}\nMax Error: {np.max(diff)}"
ax.text(0.5, 0.5, info, ha='center', va='center', fontsize=14,
        family='monospace', transform=ax.transAxes)
ax.set_title('质量指标')
ax.axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_pyramid_decompose.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n分解与重组分析:")
print(f"原始尺寸: {image.shape}, 层数: {n_levels + 1}")
print(f"PSNR: {psnr:.2f} dB, MSE: {mse:.6f}")
