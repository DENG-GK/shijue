"""
展示全局阈值在光照不均匀情况下的问题
对比全局阈值与自适应阈值的效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建光照不均匀的文档图像 =====================

def create_uneven_lighting_image():
    """创建一个光照不均匀的文档图像"""
    img = np.ones((400, 600), dtype=np.uint8) * 220

    # 添加模拟文字
    texts = ["OpenCV Tutorial", "Image Processing", "Thresholding", "Adaptive Method"]
    for i, text in enumerate(texts):
        y = 80 + i * 80
        cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 40, 2)

    # 添加不均匀光照（左边亮，右边暗）
    rows, cols = img.shape
    for j in range(cols):
        factor = 1.0 - 0.5 * (j / cols)
        img[:, j] = (img[:, j] * factor).astype(np.uint8)

    # 添加角落阴影
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt(i**2 + j**2) / np.sqrt(rows**2 + cols**2)
            img[i, j] = int(img[i, j] * (0.7 + 0.3 * dist))

    return img

# ===================== 处理 =====================

img = create_uneven_lighting_image()

# 全局阈值
_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 自适应阈值
adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 21, 10)
adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 21, 10)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像\n(光照不均匀)', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(global_thresh, cmap='gray')
axes[0, 1].set_title('全局阈值 (T=127)\n(左侧文字丢失!)', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(otsu_thresh, cmap='gray')
axes[0, 2].set_title('Otsu阈值\n(仍有问题)', fontsize=12)
axes[0, 2].axis('off')

# 直方图
axes[1, 0].hist(img.ravel(), 256, [0, 256], color='blue', alpha=0.7)
axes[1, 0].set_title('直方图\n(无明显双峰)', fontsize=12)
axes[1, 0].set_xlabel('像素值')

axes[1, 1].imshow(adaptive_mean, cmap='gray')
axes[1, 1].set_title('自适应均值\n(效果好很多!)', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(adaptive_gaussian, cmap='gray')
axes[1, 2].set_title('自适应高斯\n(最佳效果!)', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle('全局阈值 vs 自适应阈值', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('global_vs_adaptive.png', dpi=150, bbox_inches='tight')
plt.show()

print("对比分析：")
print("- 全局阈值(T=127): 左侧较暗区域的文字丢失")
print("- Otsu阈值: 仍然无法很好处理光照不均")
print("- 自适应阈值: 能够正确提取所有区域的文字")
