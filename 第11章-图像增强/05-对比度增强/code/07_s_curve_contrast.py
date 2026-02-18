"""
示例7：S曲线对比度增强
- Sigmoid函数：增强中间灰度对比度
- Tanh函数：类似效果，不同参数
- 对比不同gain/alpha参数的效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def sigmoid_contrast(image, gain=10, cutoff=0.5):
    """Sigmoid S曲线对比度增强"""
    normalized = image / 255.0
    enhanced = 1 / (1 + np.exp(-gain * (normalized - cutoff)))
    return (enhanced * 255).astype(np.uint8)


def tanh_contrast(image, alpha=3):
    """Tanh对比度增强"""
    normalized = (image / 255.0) * 2 - 1
    enhanced = np.tanh(alpha * normalized)
    return ((enhanced + 1) / 2 * 255).astype(np.uint8)


image = np.random.randint(80, 180, (300, 400), dtype=np.uint8)

gains = [5, 10, 20]
alphas = [1, 3, 5]
sig_results = [sigmoid_contrast(image, gain=g) for g in gains]
tanh_results = [tanh_contrast(image, alpha=a) for a in alphas]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('S曲线对比度增强', fontsize=14, fontweight='bold')

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('原图')
axes[0, 0].axis('off')

for i, (gain, result) in enumerate(zip(gains, sig_results)):
    axes[0, i + 1].imshow(result, cmap='gray')
    axes[0, i + 1].set_title(f'Sigmoid (gain={gain})')
    axes[0, i + 1].axis('off')

for i, (alpha, result) in enumerate(zip(alphas, tanh_results)):
    axes[1, i].imshow(result, cmap='gray')
    axes[1, i].set_title(f'Tanh (α={alpha})')
    axes[1, i].axis('off')

# S曲线
x = np.linspace(0, 1, 256)
for gain in gains:
    y = 1 / (1 + np.exp(-gain * (x - 0.5)))
    axes[1, 3].plot(x * 255, y * 255, label=f'gain={gain}', linewidth=2)
axes[1, 3].plot([0, 255], [0, 255], 'k--', alpha=0.3)
axes[1, 3].set_title('Sigmoid曲线')
axes[1, 3].legend(fontsize=8)
axes[1, 3].grid(True, alpha=0.3)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_s_curve_contrast.png'), dpi=150, bbox_inches='tight')
plt.show()
