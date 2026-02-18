"""
示例3：对比不同边缘检测算子的效果
Sobel vs Scharr vs Laplacian vs Canny
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 读取图像 =====================

# 创建测试图像
def create_test_image():
    """创建用于边缘检测测试的图像"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 绘制各种形状
    cv2.rectangle(img, (30, 30), (120, 120), 180, -1)
    cv2.circle(img, (200, 80), 50, 220, -1)
    cv2.ellipse(img, (330, 80), (40, 30), 0, 0, 360, 200, -1)

    # 三角形
    pts = np.array([[80, 180], [30, 280], [130, 280]], np.int32)
    cv2.fillPoly(img, [pts], 160)

    # 添加一些纹理
    for i in range(200, 380, 15):
        cv2.line(img, (i, 160), (i, 290), 100, 1)

    return img

img = create_test_image()

# 如果想用真实图像，取消下面的注释：
# img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# 先进行高斯模糊降噪
img_blur = cv2.GaussianBlur(img, (3, 3), 0)

print(f"图像尺寸: {img.shape}")

# ===================== 应用各种边缘检测算子 =====================

# 1. Sobel 算子
sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)
sobel = np.clip(sobel, 0, 255).astype(np.uint8)

# 2. Scharr 算子（Sobel的ksize=-1时使用Scharr）
scharr_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=-1)
scharr_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=-1)
scharr = np.sqrt(scharr_x**2 + scharr_y**2)
scharr = np.clip(scharr, 0, 255).astype(np.uint8)

# 也可以使用专门的 Scharr 函数：
# scharr_x = cv2.Scharr(img_blur, cv2.CV_64F, 1, 0)
# scharr_y = cv2.Scharr(img_blur, cv2.CV_64F, 0, 1)

# 3. Laplacian 算子
laplacian = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3)
laplacian = np.abs(laplacian)
laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)

# 4. Canny 边缘检测
canny = cv2.Canny(img_blur, 50, 150)

# ===================== 可视化对比 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

# Sobel
axes[0, 1].imshow(sobel, cmap='gray')
axes[0, 1].set_title('Sobel 算子\n(一阶导数，有平滑)', fontsize=12)
axes[0, 1].axis('off')

# Scharr
axes[0, 2].imshow(scharr, cmap='gray')
axes[0, 2].set_title('Scharr 算子\n(Sobel改进版，更精确)', fontsize=12)
axes[0, 2].axis('off')

# Laplacian
axes[1, 0].imshow(laplacian, cmap='gray')
axes[1, 0].set_title('Laplacian 算子\n(二阶导数，各向同性)', fontsize=12)
axes[1, 0].axis('off')

# Canny
axes[1, 1].imshow(canny, cmap='gray')
axes[1, 1].set_title('Canny 边缘检测\n(多阶段算法，效果最好)', fontsize=12)
axes[1, 1].axis('off')

# 说明
axes[1, 2].axis('off')
info_text = """
算子特点总结：

Sobel:
• 一阶导数算子
• 包含平滑，抗噪声
• 需要分别计算x和y方向

Scharr:
• Sobel的改进版
• 旋转对称性更好
• 精度更高

Laplacian:
• 二阶导数算子
• 一次检测所有方向
• 对噪声敏感

Canny:
• 多阶段综合算法
• 边缘细、连续
• 公认效果最好
"""
axes[1, 2].text(0.1, 0.5, info_text, fontsize=11,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('边缘检测算子对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('edge_operators_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n对比图已保存为 'edge_operators_comparison.png'")
