"""
示例7：直方图均衡化的优势展示
- 对欠曝、低对比度、雾霾图像分别进行均衡化
- 展示均衡化的自动增强效果
- 总结直方图均衡化的主要优点
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def demonstrate_advantages():
    """展示直方图均衡化的优势"""

    # 创建各种问题图像
    test_cases = {}

    # 1. 欠曝（太暗）
    dark = np.random.normal(40, 15, (200, 300)).clip(0, 255).astype(np.uint8)
    cv2.putText(dark, "Dark Image", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 80, 2)
    test_cases['欠曝图像'] = dark

    # 2. 低对比度
    low_contrast = np.random.normal(128, 20, (200, 300)).clip(0, 255).astype(np.uint8)
    cv2.rectangle(low_contrast, (50, 50), (150, 150), 100, -1)
    cv2.circle(low_contrast, (220, 100), 40, 160, -1)
    test_cases['低对比度'] = low_contrast

    # 3. 雾霾效果
    hazy = np.random.normal(150, 25, (200, 300)).clip(0, 255).astype(np.uint8)
    cv2.putText(hazy, "Hazy", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, 180, 3)
    test_cases['雾霾图像'] = hazy

    fig, axes = plt.subplots(len(test_cases), 2, figsize=(10, 4 * len(test_cases)))
    fig.suptitle('直方图均衡化的优势', fontsize=14, fontweight='bold')

    for i, (name, img) in enumerate(test_cases.items()):
        equalized = cv2.equalizeHist(img)

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'原始: {name}', fontsize=11)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(equalized, cmap='gray')
        axes[i, 1].set_title(f'均衡化: {name}', fontsize=11)
        axes[i, 1].axis('off')

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '07_advantages.png'), dpi=150, bbox_inches='tight')
    plt.show()


demonstrate_advantages()

print("\n直方图均衡化的优点：")
print("=" * 50)
print("1. 自动增强对比度，无需手动调参")
print("2. 充分利用灰度范围")
print("3. 计算简单，速度快")
print("4. 对欠曝、低对比度图像效果显著")
print("=" * 50)
