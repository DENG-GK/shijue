"""
示例4：展示噪声对边缘检测的影响
以及预处理（降噪）的重要性
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_clean_image():
    """创建干净的测试图像"""
    img = np.zeros((200, 300), dtype=np.uint8)
    img[:] = 50
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (220, 100), 40, 180, -1)
    return img

def add_noise(img, noise_level=25):
    """添加高斯噪声"""
    noise = np.random.normal(0, noise_level, img.shape).astype(np.float64)
    noisy = img.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# 创建图像
clean_img = create_clean_image()
noisy_img = add_noise(clean_img, noise_level=30)

print("图像信息：")
print(f"  干净图像噪声级别: 0")
print(f"  噪声图像添加的噪声标准差: 30")

# ===================== 边缘检测（无降噪） =====================

# 对干净图像进行边缘检测
canny_clean = cv2.Canny(clean_img, 50, 150)

# 对噪声图像直接进行边缘检测（不降噪）
canny_noisy_direct = cv2.Canny(noisy_img, 50, 150)

# ===================== 边缘检测（有降噪） =====================

# 使用不同的降噪方法
blur_gaussian = cv2.GaussianBlur(noisy_img, (5, 5), 0)
blur_median = cv2.medianBlur(noisy_img, 5)
blur_bilateral = cv2.bilateralFilter(noisy_img, 9, 75, 75)

# 降噪后再进行边缘检测
canny_gaussian = cv2.Canny(blur_gaussian, 50, 150)
canny_median = cv2.Canny(blur_median, 50, 150)
canny_bilateral = cv2.Canny(blur_bilateral, 50, 150)

# ===================== 可视化 =====================

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# 第一行：原图
axes[0, 0].imshow(clean_img, cmap='gray')
axes[0, 0].set_title('干净图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(canny_clean, cmap='gray')
axes[0, 1].set_title('干净图像的Canny结果\n（理想效果）', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(noisy_img, cmap='gray')
axes[0, 2].set_title('添加噪声后', fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(canny_noisy_direct, cmap='gray')
axes[0, 3].set_title('直接Canny（不降噪）\n（很多假边缘！）', fontsize=11)
axes[0, 3].axis('off')

# 第二行：不同降噪方法
axes[1, 0].imshow(blur_gaussian, cmap='gray')
axes[1, 0].set_title('高斯滤波降噪', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(canny_gaussian, cmap='gray')
axes[1, 1].set_title('高斯滤波后Canny', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(blur_median, cmap='gray')
axes[1, 2].set_title('中值滤波降噪', fontsize=11)
axes[1, 2].axis('off')

axes[1, 3].imshow(canny_median, cmap='gray')
axes[1, 3].set_title('中值滤波后Canny', fontsize=11)
axes[1, 3].axis('off')

# 第三行：双边滤波和总结
axes[2, 0].imshow(blur_bilateral, cmap='gray')
axes[2, 0].set_title('双边滤波降噪\n（保边效果好）', fontsize=11)
axes[2, 0].axis('off')

axes[2, 1].imshow(canny_bilateral, cmap='gray')
axes[2, 1].set_title('双边滤波后Canny\n（边缘清晰）', fontsize=11)
axes[2, 1].axis('off')

# 添加说明
axes[2, 2].axis('off')
axes[2, 3].axis('off')

summary_text = """
降噪的重要性：

噪声会产生大量假边缘，因为：
• 噪声点处灰度变化剧烈
• 梯度算子会把噪声当成边缘

解决方案：先降噪，再检测边缘

不同滤波器的效果：
• 高斯滤波：通用，可能模糊边缘
• 中值滤波：对椒盐噪声效果好
• 双边滤波：保留边缘，效果最好

建议：
对于边缘检测，优先使用双边滤波！
"""
axes[2, 2].text(0, 0.5, summary_text, fontsize=10,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('噪声对边缘检测的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('noise_effect_on_edge.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n对比图已保存为 'noise_effect_on_edge.png'")
