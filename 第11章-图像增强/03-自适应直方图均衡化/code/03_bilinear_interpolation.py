"""
示例3：理解双线性插值消除边界伪影
- 对比无插值的分块均衡化（有明显块边界）
- CLAHE的双线性插值结果（平滑过渡）
- 说明插值在CLAHE中的关键作用
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def show_interpolation_effect():
    """展示双线性插值的效果"""
    # 创建渐变测试图像
    img = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        img[i, :] = i

    # 模拟无插值的分块均衡化
    def block_equalize_no_interp(image, tile_size=64):
        result = image.copy()
        h, w = image.shape
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                tile = image[i:i + tile_size, j:j + tile_size]
                result[i:i + tile_size, j:j + tile_size] = cv2.equalizeHist(tile)
        return result

    # 无插值
    no_interp = block_equalize_no_interp(img, 64)

    # CLAHE（有插值）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    with_interp = clahe.apply(img)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('双线性插值消除块边界伪影', fontsize=14, fontweight='bold')

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('原始渐变图像', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(no_interp, cmap='gray')
    axes[1].set_title('分块均衡化（无插值）\n块边界明显可见', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(with_interp, cmap='gray')
    axes[2].set_title('CLAHE（含双线性插值）\n过渡平滑自然', fontsize=11)
    axes[2].axis('off')

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '03_bilinear_interpolation.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("双线性插值的作用：")
    print("- 消除分块边界处的不连续性")
    print("- 使结果更加平滑自然")
    print("- 这是CLAHE内部自动完成的")


show_interpolation_effect()
