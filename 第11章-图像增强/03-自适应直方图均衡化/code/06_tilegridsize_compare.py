"""
示例6：tileGridSize参数的影响
- 对比不同分块大小的效果
- 小分块更局部化，大分块接近全局
- 默认(8,8)适合大多数场景
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def compare_tile_sizes():
    """比较不同tileGridSize的效果"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 创建不同亮度的四个区域
    img[:150, :200] = np.random.normal(50, 10, (150, 200)).clip(0, 255)
    img[:150, 200:] = np.random.normal(150, 10, (150, 200)).clip(0, 255)
    img[150:, :200] = np.random.normal(100, 10, (150, 200)).clip(0, 255)
    img[150:, 200:] = np.random.normal(200, 10, (150, 200)).clip(0, 255)

    # 添加文字细节
    cv2.putText(img, "A", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 80, 3)
    cv2.putText(img, "B", (270, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 180, 3)
    cv2.putText(img, "C", (70, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, 130, 3)
    cv2.putText(img, "D", (270, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, 230, 3)
    img = img.astype(np.uint8)

    tile_sizes = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('tileGridSize参数对CLAHE效果的影响', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('原始图像', fontsize=12)
    axes[0].axis('off')

    for i, tile_size in enumerate(tile_sizes, 1):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_size)
        result = clahe.apply(img)

        axes[i].imshow(result, cmap='gray')
        axes[i].set_title(f'tileGridSize = {tile_size}', fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '06_tilegridsize_compare.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("tileGridSize 参数指南：")
    print("=" * 50)
    print("(2, 2):   很大的tiles，效果接近全局均衡化")
    print("(4, 4):   较大的tiles")
    print("(8, 8):   默认值，适合大多数情况（推荐）")
    print("(16, 16): 较小的tiles，更局部化")
    print("(32, 32): 非常小的tiles，可能产生噪声放大")
    print("=" * 50)


compare_tile_sizes()
