"""
示例8：局部对比度增强（LCE）与Unsharp Masking
- 基于局部统计量的对比度增强
- Unsharp Masking锐化增强
- 与CLAHE对比效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def local_contrast_enhancement(image, kernel_size=31, clip_limit=2.0):
    """基于局部统计量的对比度增强"""
    if kernel_size % 2 == 0:
        kernel_size += 1

    img_float = image.astype(np.float64)

    # 计算局部均值
    local_mean = cv2.blur(img_float, (kernel_size, kernel_size))

    # 计算局部方差和标准差
    local_sq_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
    local_var = local_sq_mean - local_mean ** 2
    local_std = np.sqrt(np.maximum(local_var, 0))

    # 全局统计量
    global_mean = np.mean(img_float)
    global_std = np.std(img_float)

    # 避免除零
    local_std = np.maximum(local_std, 1e-5)

    # 局部对比度增强：用全局标准差替换局部标准差
    enhanced = global_mean + (img_float - local_mean) * (global_std / local_std)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def unsharp_mask(image, sigma=1.0, strength=1.5):
    """Unsharp Masking 锐化增强"""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    enhanced = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return enhanced


# 创建测试图像（明暗分界 + 细节）
image = np.zeros((300, 400), dtype=np.uint8)
image[:, :200] = 50
image[:, 200:] = 200
cv2.circle(image, (100, 150), 50, 80, -1)
cv2.circle(image, (300, 150), 50, 170, -1)
noise = np.random.normal(0, 5, image.shape)
image = np.clip(image + noise, 0, 255).astype(np.uint8)

# LCE不同核大小
kernel_sizes = [11, 31, 51]
lce_results = [local_contrast_enhancement(image, kernel_size=ks) for ks in kernel_sizes]

# USM不同强度
strengths = [0.5, 1.0, 2.0]
usm_results = [unsharp_mask(image, sigma=2.0, strength=s) for s in strengths]

# CLAHE对比
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(image)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('局部对比度增强（LCE）与 Unsharp Masking', fontsize=14, fontweight='bold')

# 第一行：LCE
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('原图')
axes[0, 0].axis('off')
for i, (ks, result) in enumerate(zip(kernel_sizes, lce_results)):
    axes[0, i + 1].imshow(result, cmap='gray')
    axes[0, i + 1].set_title(f'LCE (k={ks})')
    axes[0, i + 1].axis('off')

# 第二行：USM + CLAHE
for i, (s, result) in enumerate(zip(strengths, usm_results)):
    axes[1, i].imshow(result, cmap='gray')
    axes[1, i].set_title(f'USM (strength={s})')
    axes[1, i].axis('off')
axes[1, 3].imshow(clahe_result, cmap='gray')
axes[1, 3].set_title('CLAHE')
axes[1, 3].axis('off')

# 第三行：直方图对比
axes[2, 0].hist(image.flatten(), bins=256, range=[0, 256], alpha=0.7)
axes[2, 0].set_title('原始直方图')
axes[2, 1].hist(lce_results[1].flatten(), bins=256, range=[0, 256], alpha=0.7, color='green')
axes[2, 1].set_title('LCE (k=31)')
axes[2, 2].hist(usm_results[1].flatten(), bins=256, range=[0, 256], alpha=0.7, color='red')
axes[2, 2].set_title('USM (s=1.0)')

# 对比度标准差比较
methods = ['原图', 'LCE', 'USM', 'CLAHE']
contrasts = [np.std(image), np.std(lce_results[1]), np.std(usm_results[1]), np.std(clahe_result)]
axes[2, 3].bar(methods, contrasts, color=['blue', 'green', 'red', 'purple'])
axes[2, 3].set_ylabel('标准差')
axes[2, 3].set_title('对比度比较')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_local_contrast.png'), dpi=150, bbox_inches='tight')
plt.show()

print("LCE：基于局部统计量，增强每个区域的局部对比度")
print("USM：通过减去模糊版本增强高频细节")
print("CLAHE：自适应直方图均衡化，局部增强效果好")
