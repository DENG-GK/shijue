"""
示例1：理解不同类型图像的直方图
- 暗图像（欠曝）、亮图像（过曝）、低对比度、高对比度、正常图像
- 通过直方图分布特征判断图像类型
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_sample_images():
    """创建不同类型的示例图像"""
    # 1. 暗图像（欠曝）
    dark = np.random.normal(50, 20, (200, 300)).clip(0, 255).astype(np.uint8)

    # 2. 亮图像（过曝）
    bright = np.random.normal(200, 20, (200, 300)).clip(0, 255).astype(np.uint8)

    # 3. 低对比度图像
    low_contrast = np.random.normal(128, 15, (200, 300)).clip(0, 255).astype(np.uint8)

    # 4. 高对比度图像
    high_contrast = np.zeros((200, 300), dtype=np.uint8)
    high_contrast[:, :150] = np.random.normal(50, 20, (200, 150)).clip(0, 255)
    high_contrast[:, 150:] = np.random.normal(200, 20, (200, 150)).clip(0, 255)

    # 5. 正常图像
    normal = np.random.normal(128, 50, (200, 300)).clip(0, 255).astype(np.uint8)

    return {
        '暗图像(欠曝)': dark,
        '亮图像(过曝)': bright,
        '低对比度': low_contrast,
        '高对比度': high_contrast,
        '正常图像': normal
    }


images = create_sample_images()

# 可视化
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('不同类型图像及其直方图', fontsize=16, fontweight='bold')

for i, (name, img) in enumerate(images.items()):
    # 显示图像
    axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, i].set_title(name, fontsize=11)
    axes[0, i].axis('off')

    # 显示直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    axes[1, i].plot(hist, color='blue')
    axes[1, i].fill_between(range(256), hist.flatten(), alpha=0.3)
    axes[1, i].set_xlim([0, 256])
    axes[1, i].set_xlabel('像素值')
    if i == 0:
        axes[1, i].set_ylabel('频率')

plt.tight_layout()

# 保存图片
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_sample_images_histogram.png'), dpi=150, bbox_inches='tight')
plt.show()

print("直方图特征分析：")
print("=" * 60)
print("暗图像:     直方图集中在左侧（低灰度值）")
print("亮图像:     直方图集中在右侧（高灰度值）")
print("低对比度:   直方图集中在中间，范围窄")
print("高对比度:   直方图分布在两端，范围宽")
print("正常图像:   直方图分布均匀，覆盖大部分范围")
