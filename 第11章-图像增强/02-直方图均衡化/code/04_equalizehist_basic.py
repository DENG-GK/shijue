"""
示例4：cv2.equalizeHist() 基本使用
- 对暗图像、亮图像、低对比度图像分别进行均衡化
- 展示均衡化前后的图像和直方图对比
- 演示cv2.equalizeHist()的标准用法
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def basic_equalization_demo():
    """直方图均衡化基本演示"""
    # 创建不同类型的测试图像
    images = {
        '暗图像(欠曝)': np.random.normal(50, 20, (200, 300)).clip(0, 255).astype(np.uint8),
        '亮图像(过曝)': np.random.normal(200, 20, (200, 300)).clip(0, 255).astype(np.uint8),
        '低对比度': np.random.normal(128, 15, (200, 300)).clip(0, 255).astype(np.uint8),
    }

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('cv2.equalizeHist() 基本使用', fontsize=14, fontweight='bold')

    for row, (name, img) in enumerate(images.items()):
        # 均衡化
        equalized = cv2.equalizeHist(img)

        # 计算直方图
        hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])

        # 显示原图
        axes[row, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[row, 0].set_title(f'原始: {name}', fontsize=10)
        axes[row, 0].axis('off')

        # 原图直方图
        axes[row, 1].plot(hist_orig)
        axes[row, 1].fill_between(range(256), hist_orig.flatten(), alpha=0.3)
        axes[row, 1].set_xlim([0, 255])
        axes[row, 1].set_title('原始直方图', fontsize=10)

        # 均衡化后
        axes[row, 2].imshow(equalized, cmap='gray', vmin=0, vmax=255)
        axes[row, 2].set_title('均衡化后', fontsize=10)
        axes[row, 2].axis('off')

        # 均衡化直方图
        axes[row, 3].plot(hist_eq, color='green')
        axes[row, 3].fill_between(range(256), hist_eq.flatten(), alpha=0.3, color='green')
        axes[row, 3].set_xlim([0, 255])
        axes[row, 3].set_title('均衡化后直方图', fontsize=10)

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '04_equalizehist_basic.png'), dpi=150, bbox_inches='tight')
    plt.show()


basic_equalization_demo()

print("cv2.equalizeHist() 使用说明：")
print("=" * 50)
print("语法: equalized = cv2.equalizeHist(gray_image)")
print("要求: 输入必须是8位单通道灰度图")
print("=" * 50)
