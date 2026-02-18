"""
示例1：Scharr算子的基本用法
学习如何使用cv2.Scharr()进行边缘检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建包含各种方向边缘的测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:] = 100

    # 垂直边缘（0°）
    cv2.rectangle(img, (30, 50), (80, 250), 200, -1)

    # 水平边缘（90°）
    cv2.rectangle(img, (110, 80), (220, 130), 200, -1)

    # 45度斜线
    cv2.line(img, (110, 170), (200, 260), 200, 8)

    # -45度斜线
    cv2.line(img, (230, 170), (320, 260), 200, 8)

    # 圆形（各方向边缘）
    cv2.circle(img, (320, 100), 45, 200, -1)

    return img

img = create_test_image()

print("测试图像信息：")
print(f"  尺寸: {img.shape}")

# ===================== 应用Scharr算子 =====================

# 方法1：使用cv2.Scharr()
scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)  # X方向
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)  # Y方向

# 方法2：等效方法 - cv2.Sobel(..., ksize=-1)
# scharr_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
# scharr_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

print(f"\nScharr结果：")
print(f"  scharr_x 范围: [{scharr_x.min():.1f}, {scharr_x.max():.1f}]")
print(f"  scharr_y 范围: [{scharr_y.min():.1f}, {scharr_y.max():.1f}]")

# ===================== 处理结果 =====================

# 取绝对值
scharr_x_abs = cv2.convertScaleAbs(scharr_x)
scharr_y_abs = cv2.convertScaleAbs(scharr_y)

# 计算梯度幅值
magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

# 计算梯度方向
direction = np.arctan2(scharr_y, scharr_x) * 180 / np.pi

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像\n(各种方向的边缘)', fontsize=11)
axes[0, 0].axis('off')

# Scharr X
axes[0, 1].imshow(scharr_x_abs, cmap='gray')
axes[0, 1].set_title('Scharr X\n(检测垂直边缘)', fontsize=11)
axes[0, 1].axis('off')

# Scharr Y
axes[0, 2].imshow(scharr_y_abs, cmap='gray')
axes[0, 2].set_title('Scharr Y\n(检测水平边缘)', fontsize=11)
axes[0, 2].axis('off')

# 梯度幅值
axes[1, 0].imshow(magnitude, cmap='gray')
axes[1, 0].set_title('梯度幅值\nsqrt(Gx^2 + Gy^2)', fontsize=11)
axes[1, 0].axis('off')

# 带颜色的方向
mask = magnitude > 30
direction_masked = np.where(mask, direction, np.nan)
im = axes[1, 1].imshow(direction_masked, cmap='hsv', vmin=-180, vmax=180)
axes[1, 1].set_title('梯度方向\n(颜色表示角度)', fontsize=11)
axes[1, 1].axis('off')
plt.colorbar(im, ax=axes[1, 1], fraction=0.046, label='角度(度)')

# 说明
axes[1, 2].axis('off')
info = """
Scharr算子使用说明：

1. 两种等效的调用方式：
   cv2.Scharr(img, ddepth, dx, dy)
   cv2.Sobel(img, ddepth, dx, dy, ksize=-1)

2. 参数说明：
   ddepth: 使用CV_64F保留负值
   dx=1, dy=0: 检测垂直边缘
   dx=0, dy=1: 检测水平边缘

3. 后处理：
   取绝对值：cv2.convertScaleAbs()
   梯度幅值：sqrt(Gx^2 + Gy^2)

4. Scharr只有3x3尺寸
   如需更大核，请使用Sobel
"""
axes[1, 2].text(0.1, 0.5, info, fontsize=10,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Scharr边缘检测基础', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('scharr_basic.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图像已保存为 'scharr_basic.png'")
