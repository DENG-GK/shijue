"""
示例4：验证完美重建
- 4种测试图像: 随机噪声/渐变/锐边/复杂
- 每种计算 max error, mean error, PSNR
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


# 测试用例
test_cases = []

# 随机噪声
random_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
test_cases.append(('随机噪声', random_img))

# 渐变图
gradient = np.zeros((128, 128, 3), dtype=np.uint8)
for i in range(128):
    gradient[:, i] = [i * 2, 128, 255 - i * 2]
test_cases.append(('渐变图', gradient))

# 锐边
edge_img = np.zeros((128, 128, 3), dtype=np.uint8)
edge_img[:, :64] = [255, 255, 255]
test_cases.append(('锐边', edge_img))

# 复杂图像
complex_img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
cv2.circle(complex_img, (64, 64), 30, (255, 0, 0), -1)
cv2.rectangle(complex_img, (80, 80), (120, 120), (0, 255, 0), -1)
test_cases.append(('复杂图像', complex_img))

results = []
for name, img in test_cases:
    lap_pyr = build_laplacian_pyramid(img, levels=4)
    recon = reconstruct_from_laplacian(lap_pyr)
    recon = np.clip(recon, 0, 255).astype(np.uint8)
    error = cv2.absdiff(img, recon)
    max_err = np.max(error)
    mean_err = np.mean(error)
    psnr = cv2.PSNR(img, recon) if max_err > 0 else float('inf')
    results.append((name, img, recon, error, max_err, mean_err, psnr))

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle('验证完美重建', fontsize=14, fontweight='bold')

for i, (name, orig, recon, error, max_err, mean_err, psnr) in enumerate(results):
    axes[i, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f'{name}\n原图')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(cv2.cvtColor(recon, cv2.COLOR_BGR2RGB))
    axes[i, 1].set_title('重建')
    axes[i, 1].axis('off')

    error_amp = cv2.convertScaleAbs(error, alpha=100)
    axes[i, 2].imshow(cv2.cvtColor(error_amp, cv2.COLOR_BGR2RGB))
    axes[i, 2].set_title(f'误差(100x)\nMax: {max_err}')
    axes[i, 2].axis('off')

    info = f"Max Error: {max_err}\nMean Error: {mean_err:.6f}\nPSNR: {psnr:.2f} dB\n完美: {'是' if max_err <= 1 else '否'}"
    axes[i, 3].text(0.5, 0.5, info, ha='center', va='center', fontsize=10,
                    family='monospace', transform=axes[i, 3].transAxes)
    axes[i, 3].set_title('指标')
    axes[i, 3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_verify_reconstruction.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n验证结果:")
print(f"{'类型':<10} {'Max Error':<12} {'Mean Error':<15} {'PSNR':<12}")
print("-" * 50)
for name, _, _, _, max_err, mean_err, psnr in results:
    print(f"{name:<10} {max_err:<12} {mean_err:<15.6f} {psnr:<12.2f}")
