"""
示例1：顶帽变换基础
提取图像中的亮细节
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建带有亮点的灰度背景图像"""
    # 创建灰色背景
    img = np.ones((200, 300), dtype=np.uint8) * 100

    # 添加亮斑点
    cv2.circle(img, (50, 50), 10, 200, -1)
    cv2.circle(img, (150, 100), 15, 220, -1)
    cv2.circle(img, (250, 60), 8, 180, -1)
    cv2.circle(img, (100, 150), 12, 210, -1)
    cv2.circle(img, (200, 160), 6, 190, -1)

    return img

# ===================== 顶帽变换 =====================

original = create_test_image()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

# 方法1：使用 morphologyEx
tophat = cv2.morphologyEx(original, cv2.MORPH_TOPHAT, kernel)

# 方法2：手动计算（原图 - 开运算）
opened = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
tophat_manual = original - opened

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('原始图像\n（灰色背景+亮斑点）', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(opened, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('开运算结果\n（亮斑点被去除）', fontsize=11)
axes[0, 1].axis('off')

axes[1, 0].imshow(tophat, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title('顶帽变换结果\n（只剩亮斑点）', fontsize=11)
axes[1, 0].axis('off')

# 增强显示
tophat_enhanced = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
axes[1, 1].imshow(tophat_enhanced, cmap='gray')
axes[1, 1].set_title('顶帽增强显示', fontsize=11)
axes[1, 1].axis('off')

plt.suptitle('顶帽变换 = 原图 - 开运算', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('tophat_basic.png', dpi=150)
plt.show()

print("两种方法结果是否一致:", np.array_equal(tophat, tophat_manual))
