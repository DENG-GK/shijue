"""
示例1：Sobel边缘检测基础用法
学习如何使用cv2.Sobel()进行边缘检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建包含各种边缘的测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 100  # 灰色背景

    # 矩形（有水平和垂直边缘）
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)

    # 圆形（有各个方向的边缘）
    cv2.circle(img, (280, 100), 50, 200, -1)

    # 斜线（有对角边缘）
    cv2.line(img, (50, 200), (180, 280), 200, 5)

    # 渐变区域
    for i in range(100):
        img[200:280, 220+i] = 100 + i

    return img

# 使用测试图像
img = create_test_image()

# 如果想用真实图像：
# img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

print("图像信息：")
print(f"  尺寸: {img.shape}")
print(f"  数据类型: {img.dtype}")

# ===================== 应用Sobel算子 =====================

# 计算x方向梯度（检测垂直边缘）
# 注意：使用CV_64F保留负值！
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# 计算y方向梯度（检测水平边缘）
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

print(f"\nSobel结果信息：")
print(f"  sobel_x 范围: [{sobel_x.min():.1f}, {sobel_x.max():.1f}]")
print(f"  sobel_y 范围: [{sobel_y.min():.1f}, {sobel_y.max():.1f}]")

# ===================== 处理结果 =====================

# 方法1：取绝对值
sobel_x_abs = np.abs(sobel_x)
sobel_y_abs = np.abs(sobel_y)

# 方法2：使用convertScaleAbs（推荐）
sobel_x_cv = cv2.convertScaleAbs(sobel_x)
sobel_y_cv = cv2.convertScaleAbs(sobel_y)

# 计算梯度幅值
# 方法1：精确计算
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

# 方法2：近似计算（更快）
magnitude_approx = cv2.addWeighted(sobel_x_cv, 0.5, sobel_y_cv, 0.5, 0)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

# x方向梯度
axes[0, 1].imshow(sobel_x_cv, cmap='gray')
axes[0, 1].set_title('Sobel X（垂直边缘）', fontsize=12)
axes[0, 1].axis('off')

# y方向梯度
axes[0, 2].imshow(sobel_y_cv, cmap='gray')
axes[0, 2].set_title('Sobel Y（水平边缘）', fontsize=12)
axes[0, 2].axis('off')

# 梯度幅值（精确）
axes[1, 0].imshow(magnitude, cmap='gray')
axes[1, 0].set_title('梯度幅值（精确）\n√(Gx²+Gy²)', fontsize=12)
axes[1, 0].axis('off')

# 梯度幅值（近似）
axes[1, 1].imshow(magnitude_approx, cmap='gray')
axes[1, 1].set_title('梯度幅值（近似）\n|Gx|+|Gy|', fontsize=12)
axes[1, 1].axis('off')

# 带颜色的方向可视化
axes[1, 2].imshow(sobel_x, cmap='RdBu', vmin=-200, vmax=200)
axes[1, 2].set_title('Sobel X（带正负）\n红=正，蓝=负', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle('Sobel边缘检测基础', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_basic.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图像已保存为 'sobel_basic.png'")
