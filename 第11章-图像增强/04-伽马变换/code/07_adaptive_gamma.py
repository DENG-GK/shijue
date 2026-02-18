"""
示例7：基于直方图的自适应伽马校正（AGC）
- 根据加权直方图分布自适应计算γ
- 利用CDF进行局部伽马调整
- 对比AGC与固定γ的效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def adaptive_gamma_correction(image, clip_limit=0.01):
    """自适应伽马校正（基于加权直方图分布）"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 计算并截断直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = np.clip(hist, 0, clip_limit)
    hist = hist / hist.sum()

    # 计算CDF
    cdf = np.cumsum(hist)

    # 计算加权均值
    intensity = np.arange(256)
    mean_intensity = np.sum(intensity * hist)

    # 根据均值计算全局伽马
    gamma = -np.log2(mean_intensity / 255.0 + 1e-8)
    gamma = np.clip(gamma, 0.1, 4.0)

    # 构建自适应LUT
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        local_gamma = gamma * (1 - cdf[i]) + 1 * cdf[i]
        lut[i] = np.power(i / 255.0, local_gamma) * 255

    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(gray, lut), gamma


# 创建测试图像（左暗右亮）
image = np.zeros((300, 400), dtype=np.uint8)
image[:, :200] = 30
image[:, 200:] = 200
for i in range(300):
    image[i, :] = np.clip(image[i, :].astype(int) + i // 3 - 50, 0, 255)
image = image.astype(np.uint8)

corrected, gamma = adaptive_gamma_correction(image)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('自适应伽马校正 (AGC)', fontsize=14, fontweight='bold')

axes[0].imshow(image, cmap='gray')
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

axes[1].imshow(corrected, cmap='gray')
axes[1].set_title(f'AGC校正 (γ≈{gamma:.2f})', fontsize=12)
axes[1].axis('off')

axes[2].hist(image.flatten(), bins=256, range=[0, 256], alpha=0.5, label='原始')
axes[2].hist(corrected.flatten(), bins=256, range=[0, 256], alpha=0.5, label='校正后')
axes[2].set_xlabel('灰度值')
axes[2].set_ylabel('频率')
axes[2].set_title('直方图对比', fontsize=12)
axes[2].legend()

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_adaptive_gamma.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"自适应伽马校正完成！(γ ≈ {gamma:.2f})")
