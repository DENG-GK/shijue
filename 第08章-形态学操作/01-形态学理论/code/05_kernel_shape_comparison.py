"""
示例5：对比不同形状的结构元素产生的效果差异
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_cross_shape():
    """创建十字形状用于测试"""
    img = np.zeros((150, 150), dtype=np.uint8)

    # 绘制十字形
    cv2.rectangle(img, (55, 20), (95, 130), 255, -1)   # 竖条
    cv2.rectangle(img, (20, 55), (130, 95), 255, -1)   # 横条

    return img

# 创建测试图像
original = create_cross_shape()

# 创建不同形状的结构元素（相同大小 9×9）
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# 对每种核执行膨胀操作
dilated_rect = cv2.dilate(original, kernel_rect)
dilated_cross = cv2.dilate(original, kernel_cross)
dilated_ellipse = cv2.dilate(original, kernel_ellipse)

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# 第一行：结构元素
axes[0, 0].set_title('原始图像', fontsize=11)
axes[0, 0].imshow(original, cmap='gray')
axes[0, 0].axis('off')

axes[0, 1].set_title('矩形核', fontsize=11)
axes[0, 1].imshow(kernel_rect, cmap='gray')
axes[0, 1].axis('off')

axes[0, 2].set_title('十字核', fontsize=11)
axes[0, 2].imshow(kernel_cross, cmap='gray')
axes[0, 2].axis('off')

axes[0, 3].set_title('椭圆核', fontsize=11)
axes[0, 3].imshow(kernel_ellipse, cmap='gray')
axes[0, 3].axis('off')

# 第二行：膨胀结果
axes[1, 0].set_title('原始图像', fontsize=11)
axes[1, 0].imshow(original, cmap='gray')
axes[1, 0].axis('off')

axes[1, 1].set_title('矩形核膨胀', fontsize=11)
axes[1, 1].imshow(dilated_rect, cmap='gray')
axes[1, 1].axis('off')

axes[1, 2].set_title('十字核膨胀', fontsize=11)
axes[1, 2].imshow(dilated_cross, cmap='gray')
axes[1, 2].axis('off')

axes[1, 3].set_title('椭圆核膨胀', fontsize=11)
axes[1, 3].imshow(dilated_ellipse, cmap='gray')
axes[1, 3].axis('off')

plt.suptitle('不同形状结构元素的膨胀效果对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('kernel_shape_comparison.png', dpi=150)
plt.show()

print("\n观察结论：")
print("• 矩形核：各方向均匀扩张，角落变直角")
print("• 十字核：只在水平和垂直方向扩张")
print("• 椭圆核：扩张后边缘更圆滑")
