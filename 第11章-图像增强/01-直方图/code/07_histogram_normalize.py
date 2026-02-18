"""
示例7：直方图归一化
- 概率归一化：直方图和为1，表示概率分布
- 最大值归一化：最大值为1，便于比较
- 对比原始直方图与两种归一化方式
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def normalize_histogram(image, norm_type='probability'):
    """
    归一化直方图

    Parameters:
    -----------
    norm_type : str
        'probability' - 归一化为概率分布（和为1）
        'max' - 归一化到[0, 1]范围（最大值为1）
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    if norm_type == 'probability':
        # 归一化为概率分布
        hist_norm = hist / hist.sum()
    else:
        # 归一化到[0, 1]
        hist_norm = hist / hist.max()

    return hist, hist_norm


# 创建测试图像
img = np.random.randint(0, 256, (300, 400), dtype=np.uint8)

hist, hist_prob = normalize_histogram(img, 'probability')
_, hist_max = normalize_histogram(img, 'max')

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('直方图归一化方法对比', fontsize=14, fontweight='bold')

axes[0].plot(hist)
axes[0].set_title('原始直方图\n（频率计数）', fontsize=11)
axes[0].set_xlabel('像素值')
axes[0].set_ylabel('计数')
axes[0].set_xlim([0, 255])

axes[1].plot(hist_prob, color='green')
axes[1].set_title('概率归一化\n（总和 = 1）', fontsize=11)
axes[1].set_xlabel('像素值')
axes[1].set_ylabel('概率')
axes[1].set_xlim([0, 255])

axes[2].plot(hist_max, color='red')
axes[2].set_title('最大值归一化\n（最大值 = 1）', fontsize=11)
axes[2].set_xlabel('像素值')
axes[2].set_ylabel('归一化值')
axes[2].set_xlim([0, 255])

plt.tight_layout()

# 保存图片
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_histogram_normalize.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"原始直方图和: {hist.sum():.0f}")
print(f"概率直方图和: {hist_prob.sum():.4f}")
print(f"最大值归一化最大值: {hist_max.max():.4f}")
