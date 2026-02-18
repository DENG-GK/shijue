"""
示例2：百分比裁剪拉伸
- 使用百分位数代替最小最大值，更鲁棒
- 忽略极端异常值（如椒盐噪声）
- 对比不同百分位裁剪的效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def percentile_contrast_stretch(image, low_percentile=2, high_percentile=98):
    """百分比裁剪拉伸"""
    p_low = np.percentile(image, low_percentile)
    p_high = np.percentile(image, high_percentile)
    clipped = np.clip(image, p_low, p_high)
    if p_high > p_low:
        stretched = (clipped - p_low) * 255 / (p_high - p_low)
    else:
        stretched = clipped
    return stretched.astype(np.uint8)


# 创建含异常值的测试图像
image = np.random.randint(60, 200, (300, 400), dtype=np.uint8)
image[0:10, 0:10] = 0      # 暗异常值
image[0:10, -10:] = 255     # 亮异常值

percentiles = [(0, 100), (1, 99), (2, 98), (5, 95)]
results = [percentile_contrast_stretch(image, low, high) for low, high in percentiles]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('百分比裁剪拉伸', fontsize=14, fontweight='bold')

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('原图')
axes[0, 0].axis('off')

for i, ((low, high), result) in enumerate(zip(percentiles, results)):
    if i < 3:
        axes[0, i + 1].imshow(result, cmap='gray')
        axes[0, i + 1].set_title(f'{low}%-{high}%')
        axes[0, i + 1].axis('off')

axes[1, 0].hist(image.flatten(), bins=256, range=[0, 256], alpha=0.7)
axes[1, 0].set_title('原始直方图')

for i, result in enumerate(results[:3]):
    axes[1, i + 1].hist(result.flatten(), bins=256, range=[0, 256], alpha=0.7)
    axes[1, i + 1].set_title(f'{percentiles[i][0]}%-{percentiles[i][1]}%')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_percentile_stretch.png'), dpi=150, bbox_inches='tight')
plt.show()

print("百分比裁剪拉伸：对异常值更鲁棒")
