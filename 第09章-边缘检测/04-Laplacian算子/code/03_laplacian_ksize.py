"""
对比Laplacian不同ksize参数的效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建带噪声的图像 =====================

def create_image_with_noise():
    """创建带轻微噪声的测试图像"""
    img = np.zeros((250, 350), dtype=np.uint8)
    img[:] = 80

    cv2.rectangle(img, (40, 40), (140, 140), 200, -1)
    cv2.circle(img, (240, 90), 50, 180, -1)
    cv2.line(img, (40, 180), (150, 230), 200, 5)

    # 添加轻微噪声
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

img = create_image_with_noise()

print("测试图像已创建（带轻微噪声）")

# ===================== 不同ksize的Laplacian =====================

ksize_list = [1, 3, 5, 7]
results = {}

for ksize in ksize_list:
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    lap_abs = cv2.convertScaleAbs(lap)
    results[ksize] = lap_abs

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像（带轻微噪声）', fontsize=11)
axes[0, 0].axis('off')

# ksize=1
axes[0, 1].imshow(results[1], cmap='gray')
axes[0, 1].set_title('ksize=1\n（标准3×3核，噪声明显）', fontsize=11)
axes[0, 1].axis('off')

# ksize=3
axes[0, 2].imshow(results[3], cmap='gray')
axes[0, 2].set_title('ksize=3\n（稍有改善）', fontsize=11)
axes[0, 2].axis('off')

# ksize=5
axes[1, 0].imshow(results[5], cmap='gray')
axes[1, 0].set_title('ksize=5\n（更多平滑）', fontsize=11)
axes[1, 0].axis('off')

# ksize=7
axes[1, 1].imshow(results[7], cmap='gray')
axes[1, 1].set_title('ksize=7\n（平滑最多，边缘变粗）', fontsize=11)
axes[1, 1].axis('off')

# 说明
axes[1, 2].axis('off')
info = """
ksize选择指南：

ksize=1:
• 使用标准3×3 Laplacian核
• 对噪声非常敏感
• 边缘最细

ksize=3:
• 使用Sobel核计算二阶导数
• 稍有平滑效果

ksize=5,7:
• 更大的核，更多平滑
• 抗噪声能力更强
• 但边缘会变粗

推荐做法：
不要依赖大ksize来抗噪
而是：先滤波 + ksize=1或3
"""
axes[1, 2].text(0.1, 0.5, info, fontsize=10,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Laplacian不同ksize对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplacian_ksize.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n对比图已保存为 'laplacian_ksize.png'")
