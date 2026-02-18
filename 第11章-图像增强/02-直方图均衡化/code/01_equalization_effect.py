"""
示例1：直方图均衡化的效果展示
- 创建低对比度图像并进行直方图均衡化
- 对比均衡化前后的图像和直方图变化
- 分析像素值范围和标准差的变化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def show_equalization_effect(image, title="图像"):
    """展示直方图均衡化前后的对比"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 直方图均衡化
    equalized = cv2.equalizeHist(gray)

    # 计算直方图
    hist_before = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'直方图均衡化效果展示 - {title}', fontsize=14, fontweight='bold')

    axes[0, 0].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('原始图像', fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].plot(hist_before, color='blue')
    axes[0, 1].fill_between(range(256), hist_before.flatten(), alpha=0.3)
    axes[0, 1].set_title('原始直方图', fontsize=11)
    axes[0, 1].set_xlim([0, 255])
    axes[0, 1].set_xlabel('像素值')
    axes[0, 1].set_ylabel('频率')

    axes[1, 0].imshow(equalized, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('均衡化后', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].plot(hist_after, color='green')
    axes[1, 1].fill_between(range(256), hist_after.flatten(), alpha=0.3, color='green')
    axes[1, 1].set_title('均衡化后直方图', fontsize=11)
    axes[1, 1].set_xlim([0, 255])
    axes[1, 1].set_xlabel('像素值')
    axes[1, 1].set_ylabel('频率')

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '01_equalization_effect.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return gray, equalized


# 创建低对比度图像
low_contrast = np.random.normal(128, 20, (300, 400)).clip(0, 255).astype(np.uint8)

# 展示效果
original, equalized = show_equalization_effect(low_contrast, "低对比度图像")

print("对比分析：")
print(f"原图 - 范围: [{original.min()}, {original.max()}], 标准差: {original.std():.1f}")
print(f"均衡化后 - 范围: [{equalized.min()}, {equalized.max()}], 标准差: {equalized.std():.1f}")
