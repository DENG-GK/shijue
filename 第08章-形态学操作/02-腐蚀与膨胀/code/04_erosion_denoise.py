"""
示例4：使用腐蚀去除二值图像中的白色噪点
这是腐蚀操作最常见的应用之一
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建带噪声的图像 =====================

def create_noisy_image():
    """创建一个带有噪点的二值图像"""
    img = np.zeros((200, 300), dtype=np.uint8)

    # 主要物体
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    cv2.circle(img, (220, 100), 40, 255, -1)

    # 添加随机噪点（模拟实际应用中的噪声）
    np.random.seed(42)
    noise_points = np.random.randint(0, 200, size=(50, 2))
    for y, x in noise_points:
        if 0 <= x < 300 and 0 <= y < 200:
            cv2.circle(img, (x + 50, y), np.random.randint(1, 4), 255, -1)

    return img

# ===================== 去噪处理 =====================

noisy = create_noisy_image()

# 使用不同大小的核进行腐蚀
kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

eroded_3 = cv2.erode(noisy, kernel_3, iterations=1)
eroded_5 = cv2.erode(noisy, kernel_5, iterations=1)

# 腐蚀后再膨胀恢复物体大小（开运算预览）
restored_3 = cv2.dilate(eroded_3, kernel_3, iterations=1)
restored_5 = cv2.dilate(eroded_5, kernel_5, iterations=1)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# 第一行：腐蚀效果
axes[0, 0].imshow(noisy, cmap='gray')
axes[0, 0].set_title('原图（带噪点）', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(eroded_3, cmap='gray')
axes[0, 1].set_title('3×3 腐蚀', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(eroded_5, cmap='gray')
axes[0, 2].set_title('5×5 腐蚀', fontsize=12)
axes[0, 2].axis('off')

# 第二行：恢复后的效果
axes[1, 0].imshow(noisy, cmap='gray')
axes[1, 0].set_title('原图（带噪点）', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(restored_3, cmap='gray')
axes[1, 1].set_title('3×3 腐蚀+膨胀', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(restored_5, cmap='gray')
axes[1, 2].set_title('5×5 腐蚀+膨胀', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle('腐蚀去除噪点（上：腐蚀效果，下：腐蚀+膨胀恢复）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('erosion_denoise.png', dpi=150)
plt.show()

print("去噪原理：")
print("• 噪点比主物体小，腐蚀时先消失")
print("• 主物体虽然变小，但不会完全消失")
print("• 腐蚀后再膨胀可以恢复主物体的大小")
print("• 这种\"先腐蚀后膨胀\"就是开运算，下一节会详细讲解")
