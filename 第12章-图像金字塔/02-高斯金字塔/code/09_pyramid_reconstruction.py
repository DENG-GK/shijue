"""
示例9：从金字塔重建图像
- 从顶层pyrUp逐层重建
- 每层重建误差分析
- PSNR计算
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), -1)
cv2.circle(image, (128, 128), 50, (255, 0, 0), -1)
cv2.putText(image, 'TEST', (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 构建高斯金字塔
n_levels = 5
gaussian = [image]
current = image
for i in range(n_levels - 1):
    current = cv2.pyrDown(current)
    gaussian.append(current)

# 从顶层重建
reconstructed = [gaussian[-1]]
for i in range(n_levels - 2, -1, -1):
    upsampled = cv2.pyrUp(reconstructed[-1])
    target_size = gaussian[i].shape[:2][::-1]
    if upsampled.shape[:2][::-1] != target_size:
        upsampled = cv2.resize(upsampled, target_size)
    reconstructed.append(upsampled)

reconstructed = reconstructed[::-1]

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle('高斯金字塔重建', fontsize=14, fontweight='bold')

for i, level in enumerate(gaussian):
    axes[0, i].imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'原始 L{i}')
    axes[0, i].axis('off')

for i, level in enumerate(reconstructed):
    axes[1, i].imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f'重建 L{i}')
    axes[1, i].axis('off')

for i, (orig, recon) in enumerate(zip(gaussian, reconstructed)):
    if orig.shape == recon.shape:
        error = cv2.absdiff(orig, recon)
        error_amp = cv2.convertScaleAbs(error, alpha=5)
        axes[2, i].imshow(cv2.cvtColor(error_amp, cv2.COLOR_BGR2RGB))
        mean_err = np.mean(error)
        axes[2, i].set_title(f'误差 L{i}\n均值: {mean_err:.1f}')
    axes[2, i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_pyramid_reconstruction.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n重建误差分析:")
for i, (orig, recon) in enumerate(zip(gaussian, reconstructed)):
    if orig.shape == recon.shape:
        error = cv2.absdiff(orig, recon)
        psnr = cv2.PSNR(orig, recon) if np.max(error) > 0 else float('inf')
        print(f"Level {i}: Mean Error={np.mean(error):.2f}, Max={np.max(error)}, PSNR={psnr:.2f} dB")
