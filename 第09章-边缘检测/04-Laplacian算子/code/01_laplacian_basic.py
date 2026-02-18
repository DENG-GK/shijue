"""
Laplacian算子的基本用法
学习如何使用cv2.Laplacian()进行边缘检测
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
    img[:] = 100

    # 矩形
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)

    # 圆形
    cv2.circle(img, (280, 100), 50, 200, -1)

    # 三角形
    pts = np.array([[80, 200], [30, 280], [130, 280]], np.int32)
    cv2.fillPoly(img, [pts], 180)

    # 渐变区域
    for i in range(100):
        img[200:280, 180+i] = 100 + int(i * 1.5)

    return img

img = create_test_image()

print("测试图像已创建")
print(f"  尺寸: {img.shape}")

# ===================== 应用Laplacian =====================

# 使用CV_64F保留负值
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=1)

print(f"\nLaplacian结果：")
print(f"  范围: [{laplacian.min():.1f}, {laplacian.max():.1f}]")
print(f"  包含负值说明检测到了从亮到暗的边缘")

# 取绝对值
laplacian_abs = np.abs(laplacian)
laplacian_abs = np.clip(laplacian_abs, 0, 255).astype(np.uint8)

# 使用convertScaleAbs
laplacian_cv = cv2.convertScaleAbs(laplacian)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

# Laplacian（带正负）
im = axes[0, 1].imshow(laplacian, cmap='RdBu', vmin=-200, vmax=200)
axes[0, 1].set_title('Laplacian（带正负）\n红=正，蓝=负', fontsize=12)
axes[0, 1].axis('off')
plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

# Laplacian绝对值
axes[1, 0].imshow(laplacian_abs, cmap='gray')
axes[1, 0].set_title('Laplacian（绝对值）', fontsize=12)
axes[1, 0].axis('off')

# 说明
axes[1, 1].axis('off')
info = """
Laplacian算子说明：

1. 函数调用：
   cv2.Laplacian(img, ddepth, ksize)

2. 参数说明：
   • ddepth: 使用CV_64F保留负值
   • ksize: 核大小（1,3,5,7）
   • ksize=1 使用标准3×3核

3. 标准Laplacian核：
   ┌────┬────┬────┐
   │ 0  │ 1  │ 0  │
   ├────┼────┼────┤
   │ 1  │ -4 │ 1  │
   ├────┼────┼────┤
   │ 0  │ 1  │ 0  │
   └────┴────┴────┘

4. 特点：
   • 一次检测所有方向边缘
   • 对噪声敏感
   • 边缘处有过零点
"""
axes[1, 1].text(0.1, 0.5, info, fontsize=10,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Laplacian边缘检测基础', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplacian_basic.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图像已保存为 'laplacian_basic.png'")
