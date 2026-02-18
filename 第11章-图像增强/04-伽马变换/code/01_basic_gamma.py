"""
示例1：基本伽马变换
- 公式：s = c * r^γ（r归一化到0-1）
- γ<1 提亮暗部，γ>1 压缩暗部
- 对比不同γ值的效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def gamma_transform(image, gamma, c=1.0):
    """
    伽马变换
    s = c * r^γ

    Parameters:
        image: 输入图像 (uint8)
        gamma: 伽马值 (γ<1 提亮, γ>1 压暗)
        c: 常数因子 (默认1.0)
    """
    normalized = image / 255.0
    transformed = c * np.power(normalized, gamma)
    transformed = np.clip(transformed * 255, 0, 255).astype(np.uint8)
    return transformed


# 创建测试图像（暗图像）
image = np.zeros((300, 400), dtype=np.uint8)
image[:] = 40
cv2.rectangle(image, (50, 50), (180, 250), 80, -1)
cv2.circle(image, (300, 150), 70, 120, -1)
cv2.ellipse(image, (200, 80), (60, 30), 30, 0, 360, 100, -1)
noise = np.random.normal(0, 5, image.shape)
image = np.clip(image + noise, 0, 255).astype(np.uint8)

# 应用不同伽马值
gamma_values = [0.3, 0.5, 1.0, 1.5, 2.5]
results = [gamma_transform(image, g) for g in gamma_values]

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('基本伽马变换 - 不同γ值效果对比', fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, (gamma, result) in enumerate(zip(gamma_values, results)):
    axes[i].imshow(result, cmap='gray', vmin=0, vmax=255)
    label = ''
    if gamma < 1:
        label = '（提亮）'
    elif gamma > 1:
        label = '（压暗）'
    else:
        label = '（原图）'
    axes[i].set_title(f'γ = {gamma} {label}', fontsize=12)
    axes[i].axis('off')

axes[5].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_basic_gamma.png'), dpi=150, bbox_inches='tight')
plt.show()

print("伽马变换公式：s = c × r^γ")
print("γ < 1: 提亮暗部，压缩亮部（适合欠曝图像）")
print("γ = 1: 无变化")
print("γ > 1: 压缩暗部，提亮亮部（适合过曝图像）")
