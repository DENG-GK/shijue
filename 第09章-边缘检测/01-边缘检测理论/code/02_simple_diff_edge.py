"""
示例2：使用简单的差分核进行边缘检测
理解最基础的边缘检测原理
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 读取图像 =====================

# 创建测试图像（或读取自己的图像）
def create_test_image():
    """创建一个包含各种边缘的测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 50  # 灰色背景

    # 白色矩形
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)

    # 白色圆形
    cv2.circle(img, (280, 100), 50, 200, -1)

    # 倾斜的线条
    cv2.line(img, (50, 200), (200, 280), 200, 5)

    # 渐变区域
    for i in range(100):
        img[200:280, 250+i] = 50 + i * 1.5

    return img

# 使用测试图像
img = create_test_image()

# 如果想用自己的图像，取消下面的注释：
# img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

print(f"图像尺寸: {img.shape}")

# ===================== 定义差分核 =====================

# 水平方向差分核（检测垂直边缘）
kernel_x = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]], dtype=np.float64)

# 垂直方向差分核（检测水平边缘）
kernel_y = np.array([[-1, -1, -1],
                     [ 0,  0,  0],
                     [ 1,  1,  1]], dtype=np.float64)

print("\n水平差分核（检测垂直边缘）:")
print(kernel_x)
print("\n垂直差分核（检测水平边缘）:")
print(kernel_y)

# ===================== 应用卷积 =====================

# 使用 filter2D 进行卷积
gradient_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
gradient_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)

# 计算梯度幅值
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# 归一化到 0-255 范围
gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

# 取绝对值并转换类型
gradient_x_abs = np.abs(gradient_x).astype(np.uint8)
gradient_y_abs = np.abs(gradient_y).astype(np.uint8)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

# 水平梯度
axes[0, 1].imshow(gradient_x_abs, cmap='gray')
axes[0, 1].set_title('水平梯度 Gx\n（检测垂直边缘）', fontsize=12)
axes[0, 1].axis('off')

# 垂直梯度
axes[0, 2].imshow(gradient_y_abs, cmap='gray')
axes[0, 2].set_title('垂直梯度 Gy\n（检测水平边缘）', fontsize=12)
axes[0, 2].axis('off')

# 梯度幅值
axes[1, 0].imshow(gradient_magnitude, cmap='gray')
axes[1, 0].set_title('梯度幅值 |∇f|\n（所有方向的边缘）', fontsize=12)
axes[1, 0].axis('off')

# 带颜色的梯度方向
axes[1, 1].imshow(gradient_x, cmap='RdBu')
axes[1, 1].set_title('水平梯度（带正负）\n红=正，蓝=负', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(gradient_y, cmap='RdBu')
axes[1, 2].set_title('垂直梯度（带正负）\n红=正，蓝=负', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle('简单差分边缘检测', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('simple_edge_detection.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n检测结果已保存为 'simple_edge_detection.png'")
