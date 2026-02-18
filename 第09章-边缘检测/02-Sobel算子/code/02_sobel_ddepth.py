"""
示例2：演示ddepth参数对Sobel结果的影响
说明为什么要使用CV_64F而不是uint8
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_edge_image():
    """创建一个有明暗两种边缘的图像"""
    img = np.zeros((200, 300), dtype=np.uint8)
    img[:] = 128  # 中等灰度背景

    # 左边：从亮到暗的边缘（导数为负）
    img[:, 50:100] = 200  # 亮区域

    # 右边：从暗到亮的边缘（导数为正）
    img[:, 200:250] = 200  # 亮区域

    return img

img = create_edge_image()

print("测试图像说明：")
print("  左边：亮→暗（x=100处，梯度为负）")
print("  右边：暗→亮（x=200处，梯度为正）")

# ===================== 不同ddepth的Sobel =====================

# 1. 使用uint8（错误做法）
# 注意：这会导致负值被截断为0！
sobel_uint8 = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)

# 2. 使用CV_64F（正确做法）
sobel_float = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# 3. 对float结果取绝对值
sobel_abs = np.abs(sobel_float)
sobel_abs = np.clip(sobel_abs, 0, 255).astype(np.uint8)

# 4. 使用convertScaleAbs
sobel_scale = cv2.convertScaleAbs(sobel_float)

print(f"\n不同方法的结果范围：")
print(f"  CV_8U直接计算: [{sobel_uint8.min()}, {sobel_uint8.max()}]")
print(f"  CV_64F计算:    [{sobel_float.min():.1f}, {sobel_float.max():.1f}]")
print(f"  取绝对值后:    [{sobel_abs.min()}, {sobel_abs.max()}]")

# ===================== 可视化对比 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像\n左:亮→暗  右:暗→亮', fontsize=11)
axes[0, 0].axis('off')

# 一维剖面图
row_data = img[100, :]
axes[0, 1].plot(row_data, 'b-', linewidth=2)
axes[0, 1].axvline(x=100, color='r', linestyle='--', alpha=0.5)
axes[0, 1].axvline(x=200, color='g', linestyle='--', alpha=0.5)
axes[0, 1].set_title('中间行的灰度值分布', fontsize=11)
axes[0, 1].set_xlabel('像素位置')
axes[0, 1].set_ylabel('灰度值')
axes[0, 1].grid(True, alpha=0.3)

# CV_8U结果（错误）
axes[0, 2].imshow(sobel_uint8, cmap='gray')
axes[0, 2].set_title('错误：CV_8U\n左边的边缘丢失了！', fontsize=11, color='red')
axes[0, 2].axis('off')

# CV_64F原始结果
im = axes[1, 0].imshow(sobel_float, cmap='RdBu', vmin=-300, vmax=300)
axes[1, 0].set_title('CV_64F原始结果\n红=正，蓝=负', fontsize=11)
axes[1, 0].axis('off')
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# 取绝对值后
axes[1, 1].imshow(sobel_abs, cmap='gray')
axes[1, 1].set_title('正确：CV_64F + 取绝对值\n两边的边缘都检测到了', fontsize=11, color='green')
axes[1, 1].axis('off')

# 对比一维剖面
row_uint8 = sobel_uint8[100, :]
row_abs = sobel_abs[100, :]
axes[1, 2].plot(row_uint8, 'r-', linewidth=2, label='CV_8U（错误）')
axes[1, 2].plot(row_abs, 'g-', linewidth=2, label='CV_64F+abs（正确）')
axes[1, 2].axvline(x=100, color='gray', linestyle='--', alpha=0.5)
axes[1, 2].axvline(x=200, color='gray', linestyle='--', alpha=0.5)
axes[1, 2].set_title('Sobel结果对比', fontsize=11)
axes[1, 2].set_xlabel('像素位置')
axes[1, 2].set_ylabel('梯度幅值')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('ddepth参数的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_ddepth.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n结论：")
print("  使用CV_8U会丢失负方向的边缘！")
print("  正确做法：使用CV_64F，然后取绝对值")
print("\n图像已保存为 'sobel_ddepth.png'")
