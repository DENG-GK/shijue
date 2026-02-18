"""
示例5：使用顶帽变换增强图像对比度
适用于增强细节较弱的图像
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建低对比度图像 =====================

def create_low_contrast_image():
    """创建低对比度的图像"""
    img = np.ones((200, 200), dtype=np.uint8) * 120  # 灰色背景

    # 添加略亮的细节（与背景对比度很低）
    cv2.rectangle(img, (30, 30), (80, 80), 140, -1)
    cv2.circle(img, (140, 60), 30, 145, -1)
    cv2.rectangle(img, (30, 120), (90, 180), 135, -1)
    cv2.circle(img, (150, 150), 35, 142, -1)

    return img

# ===================== 对比度增强 =====================

original = create_low_contrast_image()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# 顶帽变换
tophat = cv2.morphologyEx(original, cv2.MORPH_TOPHAT, kernel)

# 增强方法：原图 + 顶帽
enhanced = cv2.add(original, tophat)

# 进一步增强：增加顶帽的权重
enhanced_more = cv2.addWeighted(original, 1, tophat, 2, 0)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('原图（低对比度）', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(tophat, cmap='gray')
axes[0, 1].set_title('顶帽变换结果', fontsize=11)
axes[0, 1].axis('off')

axes[1, 0].imshow(enhanced, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title('原图 + 顶帽', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(enhanced_more, cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title('原图 + 2×顶帽\n（更强的增强）', fontsize=11)
axes[1, 1].axis('off')

plt.suptitle('顶帽变换增强对比度', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('contrast_enhancement.png', dpi=150)
plt.show()

print("增强公式：")
print("• 简单增强：enhanced = original + tophat")
print("• 强力增强：enhanced = original + k × tophat (k > 1)")
