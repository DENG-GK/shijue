"""
示例4：演示Sobel算子如何分别检测不同方向的边缘
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建有各种方向边缘的图像 =====================

def create_direction_test():
    """创建测试图像，包含不同方向的边缘"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 100

    # 垂直边缘
    cv2.rectangle(img, (30, 50), (80, 250), 200, -1)

    # 水平边缘
    cv2.rectangle(img, (120, 80), (280, 130), 200, -1)

    # 斜线（45度）
    cv2.line(img, (120, 170), (220, 270), 200, 8)

    # 斜线（-45度）
    cv2.line(img, (250, 170), (350, 270), 200, 8)

    # 圆形（各方向边缘）
    cv2.circle(img, (320, 80), 40, 200, -1)

    return img

img = create_direction_test()

# ===================== 分别检测各方向边缘 =====================

# X方向梯度（检测垂直边缘）
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_x_abs = cv2.convertScaleAbs(sobel_x)

# Y方向梯度（检测水平边缘）
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# 合并两个方向
sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

# 精确幅值
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

# 计算梯度方向
direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

print("边缘检测结果：")
print("  Sobel X: 检测垂直边缘")
print("  Sobel Y: 检测水平边缘")
print("  合并结果: 所有方向的边缘")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像\n包含垂直、水平、斜向边缘', fontsize=11)
axes[0, 0].axis('off')

# X方向
axes[0, 1].imshow(sobel_x_abs, cmap='gray')
axes[0, 1].set_title('Sobel X（|Gx|）\n检测垂直边缘', fontsize=11)
axes[0, 1].axis('off')

# Y方向
axes[0, 2].imshow(sobel_y_abs, cmap='gray')
axes[0, 2].set_title('Sobel Y（|Gy|）\n检测水平边缘', fontsize=11)
axes[0, 2].axis('off')

# 加权合并
axes[1, 0].imshow(sobel_combined, cmap='gray')
axes[1, 0].set_title('加权合并\n0.5*|Gx| + 0.5*|Gy|', fontsize=11)
axes[1, 0].axis('off')

# 精确幅值
axes[1, 1].imshow(magnitude, cmap='gray')
axes[1, 1].set_title('精确幅值\nsqrt(Gx^2 + Gy^2)', fontsize=11)
axes[1, 1].axis('off')

# 梯度方向可视化
# 只在边缘处显示方向
mask = magnitude > 30
direction_masked = np.where(mask, direction, 0)
im = axes[1, 2].imshow(direction_masked, cmap='hsv', vmin=-180, vmax=180)
axes[1, 2].set_title('梯度方向\n（颜色表示角度）', fontsize=11)
axes[1, 2].axis('off')
plt.colorbar(im, ax=axes[1, 2], fraction=0.046, label='角度 (度)')

plt.suptitle('Sobel算子检测不同方向的边缘', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_directions.png', dpi=150, bbox_inches='tight')
plt.show()

# 额外：打印不同边缘的梯度方向
print("\n梯度方向说明：")
print("  垂直边缘: Gx大, Gy约等于0 -> 方向接近0度或180度")
print("  水平边缘: Gx约等于0, Gy大 -> 方向接近90度或-90度")
print("  45度斜边: Gx约等于Gy -> 方向接近45度或-135度")
print("\n图像已保存为 'sobel_directions.png'")
