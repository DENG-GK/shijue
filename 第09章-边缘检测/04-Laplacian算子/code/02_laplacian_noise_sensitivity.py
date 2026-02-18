"""
演示Laplacian算子对噪声的敏感性
以及预处理（滤波）的重要性
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_clean_image():
    """创建干净的测试图像"""
    img = np.zeros((200, 300), dtype=np.uint8)
    img[:] = 80
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (220, 100), 40, 180, -1)
    return img

def add_noise(img, sigma=20):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float64) + noise, 0, 255)
    return noisy.astype(np.uint8)

# 创建图像
clean = create_clean_image()
noisy = add_noise(clean, sigma=25)

print("图像已创建")
print(f"  干净图像")
print(f"  噪声图像（sigma=25）")

# ===================== 直接应用Laplacian =====================

# 对干净图像
lap_clean = cv2.Laplacian(clean, cv2.CV_64F, ksize=1)
lap_clean_abs = cv2.convertScaleAbs(lap_clean)

# 对噪声图像（不预处理）
lap_noisy_direct = cv2.Laplacian(noisy, cv2.CV_64F, ksize=1)
lap_noisy_direct_abs = cv2.convertScaleAbs(lap_noisy_direct)

# ===================== 使用预滤波 =====================

# 高斯滤波后再Laplacian
noisy_gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
lap_gaussian = cv2.Laplacian(noisy_gaussian, cv2.CV_64F, ksize=1)
lap_gaussian_abs = cv2.convertScaleAbs(lap_gaussian)

# 中值滤波后再Laplacian
noisy_median = cv2.medianBlur(noisy, 5)
lap_median = cv2.Laplacian(noisy_median, cv2.CV_64F, ksize=1)
lap_median_abs = cv2.convertScaleAbs(lap_median)

# 双边滤波后再Laplacian
noisy_bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
lap_bilateral = cv2.Laplacian(noisy_bilateral, cv2.CV_64F, ksize=1)
lap_bilateral_abs = cv2.convertScaleAbs(lap_bilateral)

# ===================== 可视化 =====================

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# 第一行：原始图像
axes[0, 0].imshow(clean, cmap='gray')
axes[0, 0].set_title('干净图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(lap_clean_abs, cmap='gray')
axes[0, 1].set_title('干净图像的Laplacian\n（理想结果）', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(noisy, cmap='gray')
axes[0, 2].set_title('噪声图像', fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(lap_noisy_direct_abs, cmap='gray')
axes[0, 3].set_title('❌ 直接Laplacian\n（噪声严重！）', fontsize=11, color='red')
axes[0, 3].axis('off')

# 第二行：滤波后的图像
axes[1, 0].imshow(noisy_gaussian, cmap='gray')
axes[1, 0].set_title('高斯滤波后', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(lap_gaussian_abs, cmap='gray')
axes[1, 1].set_title('高斯滤波+Laplacian', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(noisy_median, cmap='gray')
axes[1, 2].set_title('中值滤波后', fontsize=11)
axes[1, 2].axis('off')

axes[1, 3].imshow(lap_median_abs, cmap='gray')
axes[1, 3].set_title('中值滤波+Laplacian', fontsize=11)
axes[1, 3].axis('off')

# 第三行：双边滤波和总结
axes[2, 0].imshow(noisy_bilateral, cmap='gray')
axes[2, 0].set_title('双边滤波后', fontsize=11)
axes[2, 0].axis('off')

axes[2, 1].imshow(lap_bilateral_abs, cmap='gray')
axes[2, 1].set_title('✓ 双边滤波+Laplacian\n（效果最好）', fontsize=11, color='green')
axes[2, 1].axis('off')

# 总结
axes[2, 2].axis('off')
axes[2, 3].axis('off')

summary = """
Laplacian对噪声敏感的原因：

二阶导数会放大噪声！
f'' = f(x+1) - 2f(x) + f(x-1)
噪声的二阶差分会产生很大的值

解决方案：
1. 先滤波，再Laplacian
2. 使用更大的ksize
3. 使用LoG（高斯拉普拉斯）

滤波方法推荐：
• 高斯滤波：通用，稍微模糊边缘
• 中值滤波：对椒盐噪声效果好
• 双边滤波：保边效果最好（推荐）
"""
axes[2, 2].text(0, 0.5, summary, fontsize=10,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.suptitle('Laplacian对噪声的敏感性', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplacian_noise.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n对比图已保存为 'laplacian_noise.png'")
