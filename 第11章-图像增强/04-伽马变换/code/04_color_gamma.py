"""
示例4：RGB彩色图像伽马变换
- 直接对BGR三通道应用相同的伽马变换
- 使用LUT方法高效处理
- 对比不同γ值对彩色图像的效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def gamma_transform_color(image, gamma):
    """对彩色图像应用伽马变换"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


# 创建彩色测试图像
def create_test_color_image():
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # 低亮度背景
    for i in range(300):
        for j in range(400):
            img[i, j] = [30 + int(20 * np.sin(i / 40)),
                        40 + int(20 * np.cos(j / 50)),
                        35 + int(15 * np.sin((i + j) / 60))]

    cv2.circle(img, (100, 150), 50, (60, 40, 100), -1)
    cv2.rectangle(img, (200, 80), (350, 220), (80, 120, 60), -1)
    cv2.ellipse(img, (280, 200), (50, 30), 30, 0, 360, (50, 80, 100), -1)

    return img


image = create_test_color_image()

gamma_values = [0.4, 0.7, 1.0, 1.5, 2.5]
results = [gamma_transform_color(image, g) for g in gamma_values]

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('彩色图像伽马变换', fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, (gamma, result) in enumerate(zip(gamma_values, results)):
    axes[i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f'γ = {gamma}', fontsize=12)
    axes[i].axis('off')

axes[5].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_color_gamma.png'), dpi=150, bbox_inches='tight')
plt.show()

print("彩色图像伽马变换：")
print("- 直接对所有通道应用相同的LUT")
print("- γ<1 提亮，γ>1 压暗")
print("- 注意：直接对RGB通道操作可能改变色彩饱和度")
