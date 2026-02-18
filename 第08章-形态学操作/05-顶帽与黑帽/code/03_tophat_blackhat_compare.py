"""
示例3：在同一图像上对比顶帽和黑帽的效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建混合测试图像 =====================

def create_mixed_image():
    """创建同时有亮点和暗点的图像"""
    img = np.ones((200, 300), dtype=np.uint8) * 128  # 中灰背景

    # 亮斑点
    cv2.circle(img, (50, 50), 12, 220, -1)
    cv2.circle(img, (150, 50), 10, 200, -1)
    cv2.circle(img, (250, 50), 8, 210, -1)

    # 暗斑点
    cv2.circle(img, (50, 150), 12, 50, -1)
    cv2.circle(img, (150, 150), 10, 70, -1)
    cv2.circle(img, (250, 150), 8, 60, -1)

    return img

# ===================== 顶帽和黑帽变换 =====================

original = create_mixed_image()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

tophat = cv2.morphologyEx(original, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(original, cv2.MORPH_BLACKHAT, kernel)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('原始图像\n上：亮点，下：暗点', fontsize=11)
axes[0].axis('off')

# 增强顶帽显示
tophat_vis = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
axes[1].imshow(tophat_vis, cmap='gray')
axes[1].set_title('顶帽变换\n提取亮点', fontsize=11)
axes[1].axis('off')

# 增强黑帽显示
blackhat_vis = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
axes[2].imshow(blackhat_vis, cmap='gray')
axes[2].set_title('黑帽变换\n提取暗点', fontsize=11)
axes[2].axis('off')

plt.suptitle('顶帽提取亮细节，黑帽提取暗细节', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('tophat_blackhat_compare.png', dpi=150)
plt.show()
