"""
对比Laplacian和Sobel边缘检测的效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_scene():
    """创建测试场景"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 80

    # 各种形状
    cv2.rectangle(img, (30, 30), (120, 120), 200, -1)
    cv2.circle(img, (200, 80), 45, 180, -1)

    # 线条（会产生双边缘）
    cv2.line(img, (280, 30), (380, 130), 200, 6)

    # 渐变边缘
    for i in range(80):
        img[160:260, 30+i] = 80 + int(i * 1.5)

    # 屋顶型边缘（细线条）
    cv2.line(img, (150, 200), (250, 200), 200, 2)

    # 复杂形状
    pts = np.array([[300, 160], [260, 280], [340, 280]], np.int32)
    cv2.fillPoly(img, [pts], 200)

    return img

img = create_test_scene()

# 预处理
img_blur = cv2.GaussianBlur(img, (3, 3), 0)

print("测试图像已创建")

# ===================== Sobel边缘检测 =====================

sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)

# Sobel梯度方向
sobel_dir = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

# ===================== Laplacian边缘检测 =====================

laplacian = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3)
laplacian_abs = cv2.convertScaleAbs(laplacian)

# ===================== 过零点检测 =====================

def find_zero_crossings(laplacian_img, threshold=10):
    """检测Laplacian的过零点"""
    # 过零点：相邻像素符号相反
    zero_crossings = np.zeros_like(laplacian_img, dtype=np.uint8)

    for i in range(1, laplacian_img.shape[0]-1):
        for j in range(1, laplacian_img.shape[1]-1):
            neighbors = [
                laplacian_img[i-1, j], laplacian_img[i+1, j],
                laplacian_img[i, j-1], laplacian_img[i, j+1]
            ]
            current = laplacian_img[i, j]

            # 检查是否有符号变化
            for n in neighbors:
                if current * n < 0 and abs(current - n) > threshold:
                    zero_crossings[i, j] = 255
                    break

    return zero_crossings

zero_cross = find_zero_crossings(laplacian, threshold=20)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

# Sobel
axes[0, 1].imshow(sobel_mag, cmap='gray')
axes[0, 1].set_title('Sobel（一阶导数）\n检测梯度大的区域', fontsize=12)
axes[0, 1].axis('off')

# Laplacian
axes[0, 2].imshow(laplacian_abs, cmap='gray')
axes[0, 2].set_title('Laplacian（二阶导数）\n检测二阶变化', fontsize=12)
axes[0, 2].axis('off')

# Sobel方向
mask = sobel_mag > 30
sobel_dir_masked = np.where(mask, sobel_dir, np.nan)
im = axes[1, 0].imshow(sobel_dir_masked, cmap='hsv', vmin=-180, vmax=180)
axes[1, 0].set_title('Sobel梯度方向\n（有方向信息）', fontsize=12)
axes[1, 0].axis('off')
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# Laplacian带正负
im2 = axes[1, 1].imshow(laplacian, cmap='RdBu', vmin=-100, vmax=100)
axes[1, 1].set_title('Laplacian（带正负）\n红=正，蓝=负', fontsize=12)
axes[1, 1].axis('off')
plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

# 过零点
axes[1, 2].imshow(zero_cross, cmap='gray')
axes[1, 2].set_title('Laplacian过零点\n（精确边缘位置）', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle('Sobel vs Laplacian 边缘检测对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_vs_laplacian.png', dpi=150, bbox_inches='tight')
plt.show()

# 打印对比总结
print("\nSobel vs Laplacian 对比：")
print("=" * 50)
print(f"{'特性':<20} | {'Sobel':<12} | {'Laplacian':<12}")
print("-" * 50)
print(f"{'导数阶数':<20} | {'一阶':<12} | {'二阶':<12}")
print(f"{'方向信息':<20} | {'有':<12} | {'无':<12}")
print(f"{'边缘定位':<20} | {'峰值':<12} | {'过零点':<12}")
print(f"{'噪声敏感度':<20} | {'中等':<12} | {'高':<12}")
print(f"{'双边缘问题':<20} | {'无':<12} | {'有':<12}")
print("=" * 50)

print("\n图像已保存为 'sobel_vs_laplacian.png'")
