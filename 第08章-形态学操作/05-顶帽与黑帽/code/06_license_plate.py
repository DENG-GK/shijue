"""
示例6：综合应用 - 使用顶帽和黑帽辅助车牌区域定位
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 模拟车牌图像 =====================

def create_license_plate_image():
    """模拟车牌图像（简化版）"""
    # 创建带纹理的背景（模拟车身）
    img = np.random.randint(80, 120, (200, 350), dtype=np.uint8)

    # 模糊背景
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 添加车牌区域（蓝底白字模拟）
    cv2.rectangle(img, (100, 80), (280, 140), 60, -1)  # 蓝色背景（暗）

    # 添加车牌上的字符（白色，亮）
    cv2.putText(img, 'ABC123', (110, 125), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, 200, 2)

    return img

# ===================== 处理流程 =====================

plate_img = create_license_plate_image()

# 使用不同大小的核
kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 15))

# 顶帽：提取亮字符
tophat = cv2.morphologyEx(plate_img, cv2.MORPH_TOPHAT, kernel_small)

# 黑帽：提取暗区域（车牌背景）
blackhat = cv2.morphologyEx(plate_img, cv2.MORPH_BLACKHAT, kernel_large)

# 组合：增强车牌区域
enhanced = cv2.add(tophat, blackhat)

# 二值化
_, binary = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].imshow(plate_img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(tophat, cmap='gray')
axes[0, 1].set_title('顶帽（提取亮字符）', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(blackhat, cmap='gray')
axes[0, 2].set_title('黑帽（提取暗区域）', fontsize=11)
axes[0, 2].axis('off')

axes[1, 0].imshow(enhanced, cmap='gray')
axes[1, 0].set_title('顶帽 + 黑帽', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(binary, cmap='gray')
axes[1, 1].set_title('二值化结果', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.5, '实际应用中\n会接下来做：\n• 轮廓检测\n• 区域筛选\n• 字符分割',
                ha='center', va='center', fontsize=12,
                transform=axes[1, 2].transAxes)

plt.suptitle('车牌区域提取示例', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('license_plate.png', dpi=150)
plt.show()

print("车牌检测中形态学操作的作用：")
print("• 顶帽：提取亮于背景的字符")
print("• 黑帽：提取暗于背景的车牌边框")
print("• 组合使用可以增强车牌区域的特征")
