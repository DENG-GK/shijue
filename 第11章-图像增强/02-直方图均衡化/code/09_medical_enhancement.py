"""
示例9：医学图像直方图均衡化
- 创建模拟X光图像（低对比度）
- 完整增强流程：降噪 → 均衡化 → 锐化
- 对比各步骤的效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def enhance_medical_image(image):
    """医学图像增强流程"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. 降噪
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. 直方图均衡化
    equalized = cv2.equalizeHist(denoised)

    # 3. 锐化
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(equalized, -1, kernel)

    return denoised, equalized, sharpened


def create_xray_image():
    """创建模拟X光图像"""
    img = np.zeros((400, 500), dtype=np.uint8)

    # 基础背景（低对比度）
    img[:] = 40

    # 模拟骨骼结构
    cv2.ellipse(img, (250, 200), (150, 180), 0, 0, 360, 80, -1)
    cv2.ellipse(img, (250, 200), (120, 150), 0, 0, 360, 60, -1)

    # 添加一些"病变"区域
    cv2.circle(img, (180, 150), 25, 100, -1)
    cv2.circle(img, (320, 180), 20, 95, -1)

    # 模拟低对比度和噪声
    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img


xray = create_xray_image()
denoised, equalized, sharpened = enhance_medical_image(xray)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('医学图像增强流程', fontsize=14, fontweight='bold')

axes[0, 0].imshow(xray, cmap='gray')
axes[0, 0].set_title('原始X光图像（低对比度）', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(denoised, cmap='gray')
axes[0, 1].set_title('步骤1: 高斯降噪', fontsize=11)
axes[0, 1].axis('off')

axes[1, 0].imshow(equalized, cmap='gray')
axes[1, 0].set_title('步骤2: 直方图均衡化', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(sharpened, cmap='gray')
axes[1, 1].set_title('步骤3: 锐化增强', fontsize=11)
axes[1, 1].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_medical_enhancement.png'), dpi=150, bbox_inches='tight')
plt.show()

print("医学图像增强流程：")
print("1. 降噪 (Gaussian Blur) - 减少噪声干扰")
print("2. 直方图均衡化 (Contrast Enhancement) - 增强对比度")
print("3. 锐化 (Edge Enhancement) - 突出边缘细节")
