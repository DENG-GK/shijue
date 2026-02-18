"""
示例9：医学图像CLAHE增强
- 创建模拟CT/MRI图像
- 对比全局均衡化和不同clipLimit的CLAHE效果
- CLAHE能更好地保留局部细节
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def enhance_medical_image_clahe(image, clip_limit=2.0):
    """使用CLAHE增强医学图像"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


def create_medical_scan():
    """创建模拟CT/MRI图像"""
    img = np.zeros((400, 400), dtype=np.uint8)
    img[:] = 30

    # 模拟组织结构
    cv2.ellipse(img, (200, 200), (150, 120), 0, 0, 360, 60, -1)
    cv2.ellipse(img, (200, 200), (100, 80), 0, 0, 360, 50, -1)

    # 模拟病变区域
    cv2.circle(img, (150, 180), 25, 70, -1)
    cv2.circle(img, (260, 160), 20, 65, -1)
    cv2.ellipse(img, (200, 240), (15, 10), 30, 0, 360, 75, -1)

    noise = np.random.normal(0, 3, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img


medical_img = create_medical_scan()

global_eq = cv2.equalizeHist(medical_img)
clahe_result = enhance_medical_image_clahe(medical_img, clip_limit=2.0)
clahe_strong = enhance_medical_image_clahe(medical_img, clip_limit=4.0)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('医学图像CLAHE增强', fontsize=14, fontweight='bold')

axes[0, 0].imshow(medical_img, cmap='gray')
axes[0, 0].set_title('原始医学图像', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(global_eq, cmap='gray')
axes[0, 1].set_title('全局均衡化\n（过度增强）', fontsize=11)
axes[0, 1].axis('off')

axes[1, 0].imshow(clahe_result, cmap='gray')
axes[1, 0].set_title('CLAHE (clipLimit=2.0)\n（推荐）', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(clahe_strong, cmap='gray')
axes[1, 1].set_title('CLAHE (clipLimit=4.0)\n（较强）', fontsize=11)
axes[1, 1].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_medical_clahe.png'), dpi=150, bbox_inches='tight')
plt.show()

print("医学图像增强建议：")
print("- clipLimit: 通常使用2.0-4.0")
print("- tileGridSize: (8, 8)适合大多数情况")
print("- CLAHE能更好地保留局部细节")
