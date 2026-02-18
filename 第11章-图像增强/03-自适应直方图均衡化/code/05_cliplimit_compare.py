"""
示例5：clipLimit参数的影响
- 对比clipLimit从1.0到40.0的效果变化
- 较小值：增强弱但稳定；较大值：增强强接近全局均衡化
- 给出不同场景的推荐值
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def compare_clip_limits():
    """比较不同clipLimit的效果"""
    img = np.random.normal(80, 30, (250, 350)).clip(0, 255).astype(np.uint8)
    cv2.rectangle(img, (50, 30), (150, 200), 120, -1)
    cv2.circle(img, (250, 120), 50, 60, -1)

    clip_limits = [1.0, 2.0, 4.0, 8.0, 16.0, 40.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('clipLimit参数对CLAHE效果的影响', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, clip in enumerate(clip_limits):
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        result = clahe.apply(img)

        axes[i].imshow(result, cmap='gray')
        axes[i].set_title(f'clipLimit = {clip}', fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '05_cliplimit_compare.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("clipLimit 参数指南：")
    print("=" * 50)
    print("1.0:   很弱的增强，接近原图")
    print("2.0:   适中的增强（推荐起始值）")
    print("4.0:   较强的增强")
    print("8.0+:  非常强的增强")
    print("40.0:  接近普通直方图均衡化")
    print("=" * 50)


compare_clip_limits()
