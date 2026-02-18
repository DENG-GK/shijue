"""
示例11：医学图像伽马增强
- 伽马变换提亮X光图像暗部细节
- 伽马 + CLAHE组合增强
- 对比不同γ值和组合方法的效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def enhance_medical_image(image, gamma=0.7, clahe_clip=2.0):
    """医学图像伽马+CLAHE增强"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    gamma_corrected = cv2.LUT(image, table)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gamma_corrected)

    return enhanced, gamma_corrected


def create_xray_simulation():
    """创建模拟X光图像"""
    image = np.zeros((300, 300), dtype=np.uint8)
    image[:] = 30

    cv2.ellipse(image, (150, 150), (120, 80), 0, 0, 360, 180, -1)
    cv2.ellipse(image, (150, 150), (100, 60), 0, 0, 360, 60, -1)

    cv2.circle(image, (120, 140), 30, 100, -1)
    cv2.circle(image, (180, 140), 30, 100, -1)

    cv2.line(image, (80, 230), (220, 230), 150, 3)
    cv2.line(image, (120, 210), (120, 270), 120, 2)
    cv2.line(image, (180, 210), (180, 270), 120, 2)

    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image


xray = create_xray_simulation()

# 不同γ值
gamma_values = [0.5, 0.7, 1.0, 1.5]
results = [enhance_medical_image(xray, gamma=g) for g in gamma_values]

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('医学图像伽马增强', fontsize=14, fontweight='bold')

# 仅伽马
axes[0, 0].imshow(xray, cmap='gray')
axes[0, 0].set_title('原始X光', fontsize=10)
axes[0, 0].axis('off')

for i, (gamma, (enhanced, gamma_only)) in enumerate(zip(gamma_values[:-1], results[:-1])):
    axes[0, i + 1].imshow(gamma_only, cmap='gray')
    axes[0, i + 1].set_title(f'伽马 γ={gamma}', fontsize=10)
    axes[0, i + 1].axis('off')

# 伽马+CLAHE
axes[1, 0].imshow(xray, cmap='gray')
axes[1, 0].set_title('原始X光', fontsize=10)
axes[1, 0].axis('off')

for i, (gamma, (enhanced, _)) in enumerate(zip(gamma_values[:-1], results[:-1])):
    axes[1, i + 1].imshow(enhanced, cmap='gray')
    axes[1, i + 1].set_title(f'γ={gamma}+CLAHE', fontsize=10)
    axes[1, i + 1].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '11_medical_gamma.png'), dpi=150, bbox_inches='tight')
plt.show()

print("医学图像增强建议：")
print("- γ=0.5~0.7: 提亮暗部组织细节")
print("- CLAHE进一步增强局部对比度")
print("- 组合使用效果更佳")
