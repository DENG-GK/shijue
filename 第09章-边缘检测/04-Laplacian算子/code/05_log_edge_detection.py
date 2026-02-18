"""
LoG（Laplacian of Gaussian）边缘检测
高斯滤波+Laplacian的组合方法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_noisy_image():
    """创建带噪声的测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 80

    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (280, 100), 50, 180, -1)
    cv2.ellipse(img, (100, 220), (60, 40), 0, 0, 360, 200, -1)

    # 添加噪声
    noise = np.random.normal(0, 15, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

img = create_noisy_image()

print("带噪声的测试图像已创建")

# ===================== 不同方法对比 =====================

# 方法1：直接Laplacian
lap_direct = cv2.Laplacian(img, cv2.CV_64F, ksize=1)
lap_direct_abs = cv2.convertScaleAbs(lap_direct)

# 方法2：先高斯滤波，再Laplacian（LoG）
def apply_log(img, sigma=1.0):
    """应用LoG（Laplacian of Gaussian）"""
    # 根据sigma计算核大小
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    # 高斯滤波
    gaussian = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    # Laplacian
    laplacian = cv2.Laplacian(gaussian, cv2.CV_64F, ksize=1)

    return laplacian, gaussian

# 不同sigma的LoG
sigmas = [1.0, 2.0, 3.0]
log_results = {}

for sigma in sigmas:
    lap, gauss = apply_log(img, sigma)
    log_results[sigma] = {
        'gaussian': gauss,
        'laplacian': cv2.convertScaleAbs(lap),
        'raw': lap
    }

# 方法3：使用OpenCV的Laplacian with large ksize
lap_ksize5 = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
lap_ksize5_abs = cv2.convertScaleAbs(lap_ksize5)

# ===================== 可视化 =====================

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# 第一行：原图和直接Laplacian
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始噪声图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(lap_direct_abs, cmap='gray')
axes[0, 1].set_title('❌ 直接Laplacian\n（噪声严重）', fontsize=11, color='red')
axes[0, 1].axis('off')

axes[0, 2].imshow(lap_ksize5_abs, cmap='gray')
axes[0, 2].set_title('Laplacian (ksize=5)\n（稍有改善）', fontsize=11)
axes[0, 2].axis('off')

# 说明
axes[0, 3].axis('off')
info = """
LoG = Laplacian of Gaussian

原理：
1. 先用高斯滤波平滑噪声
2. 再用Laplacian检测边缘

LoG的优点：
• 抑制噪声
• 保留边缘

sigma的影响：
• sigma小：保留更多细节
• sigma大：更多平滑，边缘更粗
"""
axes[0, 3].text(0.1, 0.5, info, fontsize=10,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# 第二行：不同sigma的高斯滤波结果
for i, sigma in enumerate(sigmas):
    axes[1, i].imshow(log_results[sigma]['gaussian'], cmap='gray')
    axes[1, i].set_title(f'高斯滤波 (σ={sigma})', fontsize=11)
    axes[1, i].axis('off')

axes[1, 3].axis('off')
axes[1, 3].text(0.5, 0.5, 'σ越大\n图像越模糊', fontsize=12,
                ha='center', va='center', fontfamily='SimHei')

# 第三行：不同sigma的LoG结果
for i, sigma in enumerate(sigmas):
    axes[2, i].imshow(log_results[sigma]['laplacian'], cmap='gray')
    axes[2, i].set_title(f'✓ LoG (σ={sigma})', fontsize=11, color='green')
    axes[2, i].axis('off')

axes[2, 3].axis('off')
axes[2, 3].text(0.5, 0.5, 'σ越大\n边缘越粗\n噪声越少', fontsize=12,
                ha='center', va='center', fontfamily='SimHei')

plt.suptitle('LoG（Laplacian of Gaussian）边缘检测', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplacian_of_gaussian.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLoG对比图已保存为 'laplacian_of_gaussian.png'")
