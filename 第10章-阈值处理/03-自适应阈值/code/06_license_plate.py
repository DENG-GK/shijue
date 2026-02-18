"""
车牌识别预处理
使用CLAHE增强对比度 + 自适应阈值进行二值化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 车牌预处理函数 =====================

def preprocess_license_plate(image):
    """车牌图像预处理"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 自适应阈值
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 4
    )

    # 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return gray, enhanced, binary

# ===================== 创建模拟车牌图像 =====================

def create_license_plate():
    img = np.ones((120, 400), dtype=np.uint8) * 200

    # 车牌背景渐变（模拟光照）
    for j in range(400):
        factor = 0.7 + 0.3 * np.abs(np.sin(j / 80))
        img[:, j] = (img[:, j] * factor).astype(np.uint8)

    # 添加车牌号
    cv2.putText(img, "A-12345", (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, 30, 6)

    # 添加噪声
    noise = np.random.normal(0, 15, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# ===================== 处理车牌 =====================

plate_img = create_license_plate()
gray, enhanced, binary = preprocess_license_plate(plate_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].imshow(plate_img, cmap='gray')
axes[0, 0].set_title('原始车牌\n(光照不均匀)', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(enhanced, cmap='gray')
axes[0, 1].set_title('CLAHE增强\n(对比度提升)', fontsize=11)
axes[0, 1].axis('off')

axes[1, 0].imshow(binary, cmap='gray')
axes[1, 0].set_title('自适应阈值\n(适合OCR识别)', fontsize=11)
axes[1, 0].axis('off')

# 直方图对比
axes[1, 1].hist(plate_img.ravel(), 256, [0, 256], alpha=0.5, label='原始')
axes[1, 1].hist(enhanced.ravel(), 256, [0, 256], alpha=0.5, label='增强后')
axes[1, 1].set_title('直方图对比', fontsize=11)
axes[1, 1].legend()
axes[1, 1].set_xlabel('像素值')

plt.suptitle('车牌识别预处理', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('license_plate.png', dpi=150, bbox_inches='tight')
plt.show()

print("车牌预处理完成！")
print("处理流程: 灰度化 → CLAHE增强 → 自适应阈值 → 形态学清理")
