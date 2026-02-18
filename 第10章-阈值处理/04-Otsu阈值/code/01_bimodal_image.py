"""
理想的双峰直方图
展示Otsu阈值最适合的图像类型
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建双峰分布图像 =====================

def create_bimodal_image():
    """创建具有明显双峰分布的图像"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 左半部分：暗色（背景）
    img[:, :200] = np.random.normal(60, 15, (300, 200)).clip(0, 255)

    # 右半部分：亮色（前景）
    img[:, 200:] = np.random.normal(190, 15, (300, 200)).clip(0, 255)

    return img.astype(np.uint8)

# ===================== 处理 =====================

bimodal_img = create_bimodal_image()

# 计算直方图
hist = cv2.calcHist([bimodal_img], [0], None, [256], [0, 256]).flatten()

# 应用Otsu阈值
otsu_thresh, binary = cv2.threshold(bimodal_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(bimodal_img, cmap='gray')
axes[0].set_title('原始图像\n(双峰分布)', fontsize=11)
axes[0].axis('off')

axes[1].plot(hist)
axes[1].axvline(x=otsu_thresh, color='r', linestyle='--', linewidth=2, label=f'Otsu T={otsu_thresh:.0f}')
axes[1].fill_between(range(256), hist, alpha=0.3)
axes[1].set_title('直方图\n(两个明显的峰)', fontsize=11)
axes[1].set_xlabel('像素值')
axes[1].set_ylabel('频率')
axes[1].legend()

axes[2].imshow(binary, cmap='gray')
axes[2].set_title(f'Otsu结果\n(T={otsu_thresh:.0f})', fontsize=11)
axes[2].axis('off')

plt.suptitle('Otsu阈值 - 双峰图像', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('bimodal_image.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Otsu自动选择的阈值: {otsu_thresh:.0f}")
print("此阈值正好位于两个峰之间的谷底位置")
