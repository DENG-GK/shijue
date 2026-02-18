"""
使用Laplacian进行图像锐化
锐化 = 原图 - α × Laplacian
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建或读取图像 =====================

def create_blur_image():
    """创建一个稍微模糊的测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 绘制场景
    img[:] = 120

    # 建筑
    cv2.rectangle(img, (50, 100), (150, 250), 80, -1)
    cv2.rectangle(img, (70, 120), (90, 160), 180, -1)  # 窗户
    cv2.rectangle(img, (110, 120), (130, 160), 180, -1)  # 窗户
    cv2.rectangle(img, (90, 190), (120, 250), 60, -1)  # 门

    # 树
    cv2.rectangle(img, (220, 180), (240, 250), 100, -1)  # 树干
    cv2.circle(img, (230, 140), 50, 70, -1)  # 树冠

    # 太阳
    cv2.circle(img, (330, 60), 30, 200, -1)

    # 添加轻微模糊
    img = cv2.GaussianBlur(img, (5, 5), 1)

    return img

img = create_blur_image()

print("模糊图像已创建")

# ===================== 图像锐化 =====================

def laplacian_sharpening(img, alpha=1.0):
    """
    使用Laplacian进行图像锐化
    锐化图像 = 原图 - alpha × Laplacian

    注意：这里用减法是因为我们使用的Laplacian核中心为负
    如果核中心为正，则用加法
    """
    # 计算Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)

    # 锐化：原图 - alpha × Laplacian
    sharpened = img.astype(np.float64) - alpha * laplacian

    # 裁剪到有效范围
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened, laplacian

# 不同alpha值的锐化效果
alphas = [0.5, 1.0, 1.5, 2.0]
sharpened_results = {}

for alpha in alphas:
    sharp, lap = laplacian_sharpening(img, alpha)
    sharpened_results[alpha] = sharp

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像（稍模糊）', fontsize=12)
axes[0, 0].axis('off')

# Laplacian
_, laplacian = laplacian_sharpening(img, 1.0)
axes[0, 1].imshow(cv2.convertScaleAbs(laplacian), cmap='gray')
axes[0, 1].set_title('Laplacian（边缘信息）', fontsize=12)
axes[0, 1].axis('off')

# alpha=1.0的锐化结果
axes[0, 2].imshow(sharpened_results[1.0], cmap='gray')
axes[0, 2].set_title('锐化结果 (α=1.0)', fontsize=12)
axes[0, 2].axis('off')

# 不同alpha对比
axes[1, 0].imshow(sharpened_results[0.5], cmap='gray')
axes[1, 0].set_title('α=0.5（轻微锐化）', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(sharpened_results[1.5], cmap='gray')
axes[1, 1].set_title('α=1.5（较强锐化）', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(sharpened_results[2.0], cmap='gray')
axes[1, 2].set_title('α=2.0（过度锐化）', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle('Laplacian图像锐化', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplacian_sharpening.png', dpi=150, bbox_inches='tight')
plt.show()

# 绘制一维剖面对比
fig, ax = plt.subplots(figsize=(12, 5))

row = 140
ax.plot(img[row, :], 'b-', linewidth=2, label='原图')
ax.plot(sharpened_results[1.0][row, :], 'r-', linewidth=2, label='锐化后 (α=1.0)')
ax.set_xlabel('像素位置', fontsize=12)
ax.set_ylabel('灰度值', fontsize=12)
ax.set_title(f'第{row}行的灰度值对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('laplacian_sharpening_profile.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n锐化效果图已保存")
print("\n锐化原理：")
print("  锐化图像 = 原图 - α × Laplacian")
print("  • α < 1: 轻微锐化")
print("  • α = 1: 标准锐化")
print("  • α > 1: 强烈锐化（可能过度）")
