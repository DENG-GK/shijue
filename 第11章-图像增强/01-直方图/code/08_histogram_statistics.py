"""
示例8：从直方图提取统计信息
- 计算均值、标准差、方差、众数、中位数
- 计算偏度和峰度
- 对比不同特征图像的统计量差异
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def histogram_statistics(image):
    """从直方图计算统计量"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # 归一化为概率分布
    prob = hist / hist.sum()

    # 像素值
    values = np.arange(256)

    # 统计量
    mean = np.sum(values * prob)
    variance = np.sum(((values - mean) ** 2) * prob)
    std = np.sqrt(variance)

    # 众数（最常见的灰度值）
    mode = np.argmax(hist)

    # 中位数
    cumsum = np.cumsum(prob)
    median = np.searchsorted(cumsum, 0.5)

    # 偏度和峰度
    skewness = np.sum(((values - mean) / std) ** 3 * prob)
    kurtosis = np.sum(((values - mean) / std) ** 4 * prob) - 3

    stats = {
        'mean': mean,
        'std': std,
        'variance': variance,
        'mode': mode,
        'median': median,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'min': gray.min(),
        'max': gray.max()
    }

    return hist, stats


# 创建不同特征的图像
images = {
    '暗图像': np.random.normal(60, 20, (200, 300)).clip(0, 255).astype(np.uint8),
    '亮图像': np.random.normal(200, 20, (200, 300)).clip(0, 255).astype(np.uint8),
    '高对比度': np.random.normal(128, 60, (200, 300)).clip(0, 255).astype(np.uint8),
    '低对比度': np.random.normal(128, 15, (200, 300)).clip(0, 255).astype(np.uint8),
}

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('不同图像的直方图统计量', fontsize=16, fontweight='bold')

for i, (name, img) in enumerate(images.items()):
    hist, stats = histogram_statistics(img)

    # 显示图像
    axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, i].set_title(name, fontsize=11)
    axes[0, i].axis('off')

    # 显示直方图并标注统计量
    axes[1, i].plot(hist, color='blue', linewidth=1)
    axes[1, i].fill_between(range(256), hist, alpha=0.2)
    axes[1, i].axvline(stats['mean'], color='red', linestyle='--', label=f"均值={stats['mean']:.0f}")
    axes[1, i].axvline(stats['median'], color='green', linestyle='--', label=f"中位数={stats['median']}")
    axes[1, i].axvline(stats['mode'], color='orange', linestyle='--', label=f"众数={stats['mode']}")
    axes[1, i].set_xlim([0, 255])
    axes[1, i].set_xlabel('像素值')
    if i == 0:
        axes[1, i].set_ylabel('频率')
    axes[1, i].legend(fontsize=8)

plt.tight_layout()

# 保存图片
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_histogram_statistics.png'), dpi=150, bbox_inches='tight')
plt.show()

# 分析每个图像
print("图像直方图统计量分析：")
print("=" * 80)
print(f"{'图像':10s} {'均值':>8s} {'标准差':>8s} {'众数':>6s} {'中位数':>8s} {'偏度':>10s}")
print("=" * 80)

for name, img in images.items():
    hist, stats = histogram_statistics(img)
    print(f"{name:10s} {stats['mean']:8.1f} {stats['std']:8.1f} "
          f"{stats['mode']:6d} {stats['median']:8d} {stats['skewness']:10.2f}")

print("=" * 80)
print("\n统计量含义：")
print("- Mean: 平均灰度值，反映整体亮度")
print("- Std: 标准差，反映对比度")
print("- Mode: 众数，最常见的灰度值")
print("- Median: 中位数，中间灰度值")
print("- Skewness: 偏度，正值表示右偏（暗图像），负值表示左偏（亮图像）")
