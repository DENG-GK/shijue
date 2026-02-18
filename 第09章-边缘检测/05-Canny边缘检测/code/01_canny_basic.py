"""
Canny边缘检测的基本用法
学习如何使用cv2.Canny()进行边缘检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 80

    # 各种形状
    cv2.rectangle(img, (40, 40), (140, 140), 200, -1)
    cv2.circle(img, (260, 90), 50, 180, -1)
    cv2.ellipse(img, (320, 200), (50, 30), 30, 0, 360, 200, -1)

    # 线条
    cv2.line(img, (40, 180), (150, 260), 200, 4)

    # 三角形
    pts = np.array([[200, 160], [160, 280], [240, 280]], np.int32)
    cv2.fillPoly(img, [pts], 180)

    return img

img = create_test_image()

print("测试图像已创建")
print(f"  尺寸: {img.shape}")

# ===================== 应用Canny边缘检测 =====================

# 基本用法
edges = cv2.Canny(img, 50, 150)

print(f"\nCanny结果：")
print(f"  输出类型: {edges.dtype}")
print(f"  像素值: 0 或 255（二值图）")

# ===================== 与其他方法对比 =====================

# 高斯模糊后再Canny（标准做法）
img_blur = cv2.GaussianBlur(img, (5, 5), 1.4)
edges_blur = cv2.Canny(img_blur, 50, 150)

# Sobel
sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)
sobel = np.clip(sobel, 0, 255).astype(np.uint8)

# Laplacian
laplacian = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

# Canny（不预处理）
axes[0, 1].imshow(edges, cmap='gray')
axes[0, 1].set_title('Canny (50, 150)\n直接处理', fontsize=12)
axes[0, 1].axis('off')

# Canny（高斯模糊后）
axes[0, 2].imshow(edges_blur, cmap='gray')
axes[0, 2].set_title('Canny (50, 150)\n高斯模糊后', fontsize=12)
axes[0, 2].axis('off')

# Sobel
axes[1, 0].imshow(sobel, cmap='gray')
axes[1, 0].set_title('Sobel\n（边缘较粗）', fontsize=12)
axes[1, 0].axis('off')

# Laplacian
axes[1, 1].imshow(laplacian, cmap='gray')
axes[1, 1].set_title('Laplacian\n（边缘不连续）', fontsize=12)
axes[1, 1].axis('off')

# 对比说明
axes[1, 2].axis('off')
info = """
Canny边缘检测的优势：

✓ 边缘细（单像素宽）
✓ 边缘连续
✓ 噪声抑制好
✓ 定位准确

使用方法：
edges = cv2.Canny(img, 50, 150)

参数说明：
• 50: 低阈值
• 150: 高阈值
• 推荐：高阈值 = 2~3 × 低阈值

标准处理流程：
1. 高斯模糊（可选）
2. Canny边缘检测
"""
axes[1, 2].text(0.1, 0.5, info, fontsize=10,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.suptitle('Canny边缘检测基础', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('canny_basic.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图像已保存为 'canny_basic.png'")
