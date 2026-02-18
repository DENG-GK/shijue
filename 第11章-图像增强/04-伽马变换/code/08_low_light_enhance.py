"""
示例8：低光图像增强
- 在LAB空间对L通道进行伽马变换
- 可选的降噪处理
- 对比不同γ值的增强效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def enhance_low_light_image(image, gamma=0.4, denoise=True):
    """低光照图像增强"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 伽马变换L通道
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    l_enhanced = cv2.LUT(l, table)

    # 降噪（可选）
    if denoise:
        l_enhanced = cv2.fastNlMeansDenoising(l_enhanced, None, 10, 7, 21)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced


# 创建模拟低光照图像
def create_low_light_image():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # 暗背景
    for i in range(300):
        for j in range(400):
            img[i, j] = [15 + int(10 * np.sin(i / 40)),
                        20 + int(8 * np.cos(j / 50)),
                        12 + int(6 * np.sin((i + j) / 60))]

    # 一些暗物体
    cv2.rectangle(img, (50, 50), (180, 200), (40, 50, 30), -1)
    cv2.circle(img, (300, 150), 60, (35, 45, 55), -1)
    cv2.ellipse(img, (200, 250), (80, 30), 0, 0, 360, (50, 40, 35), -1)

    # 添加噪声
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


low_light = create_low_light_image()

# 不同γ值增强
gamma_values = [0.3, 0.5, 0.7]
enhanced_images = [enhance_low_light_image(low_light, g) for g in gamma_values]

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('低光照图像伽马增强', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(low_light, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始低光照图像', fontsize=12)
axes[0, 0].axis('off')

for i, (gamma, enhanced) in enumerate(zip(gamma_values, enhanced_images)):
    row, col = (i + 1) // 2, (i + 1) % 2
    axes[row, col].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f'增强 γ={gamma}', fontsize=12)
    axes[row, col].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_low_light_enhance.png'), dpi=150, bbox_inches='tight')
plt.show()

print("低光照增强建议：")
print("- γ=0.3: 强力提亮，适合非常暗的图像")
print("- γ=0.5: 中等提亮，效果自然")
print("- γ=0.7: 轻微提亮，适合稍暗的图像")
print("- 建议在LAB空间处理以保持色彩")
