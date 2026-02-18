"""
示例2：黑帽变换基础
提取图像中的暗细节
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建带有暗点的白色背景图像"""
    # 创建浅灰色背景
    img = np.ones((200, 300), dtype=np.uint8) * 180

    # 添加暗斑点
    cv2.circle(img, (50, 50), 10, 80, -1)
    cv2.circle(img, (150, 100), 15, 60, -1)
    cv2.circle(img, (250, 60), 8, 100, -1)
    cv2.circle(img, (100, 150), 12, 70, -1)
    cv2.circle(img, (200, 160), 6, 90, -1)

    return img

# ===================== 黑帽变换 =====================

original = create_test_image()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

# 方法1：使用 morphologyEx
blackhat = cv2.morphologyEx(original, cv2.MORPH_BLACKHAT, kernel)

# 方法2：手动计算（闭运算 - 原图）
closed = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)
blackhat_manual = closed - original

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('原始图像\n（浅色背景+暗斑点）', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(closed, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('闭运算结果\n（暗斑点被填上）', fontsize=11)
axes[0, 1].axis('off')

axes[1, 0].imshow(blackhat, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title('黑帽变换结果\n（只剩暗斑点）', fontsize=11)
axes[1, 0].axis('off')

# 增强显示
blackhat_enhanced = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
axes[1, 1].imshow(blackhat_enhanced, cmap='gray')
axes[1, 1].set_title('黑帽增强显示', fontsize=11)
axes[1, 1].axis('off')

plt.suptitle('黑帽变换 = 闭运算 - 原图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('blackhat_basic.png', dpi=150)
plt.show()

print("两种方法结果是否一致:", np.array_equal(blackhat, blackhat_manual))
