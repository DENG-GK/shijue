"""
医学图像分割（模拟）
使用CLAHE增强 + Otsu阈值 + 形态学后处理进行细胞检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 医学图像分割函数 =====================

def segment_medical_image(image):
    """医学图像分割流程"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. 预处理：高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # 3. Otsu阈值分割
    thresh, binary = cv2.threshold(enhanced, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 形态学后处理
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. 提取轮廓
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制结果
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return {
        'gray': gray,
        'enhanced': enhanced,
        'binary': binary,
        'cleaned': cleaned,
        'result': result,
        'threshold': thresh,
        'contours': len(contours)
    }

# ===================== 创建模拟医学图像 =====================

def create_medical_image():
    """创建模拟细胞图像"""
    img = np.random.normal(100, 20, (400, 500)).astype(np.uint8)

    cells = [
        (100, 100, 30), (200, 150, 25), (350, 120, 35),
        (150, 280, 28), (300, 300, 32), (420, 250, 26)
    ]

    for x, y, r in cells:
        cv2.circle(img, (x, y), r, 200, -1)
        cv2.circle(img, (x, y), r // 3, 230, -1)

    return img

# ===================== 运行分割 =====================

medical_img = create_medical_image()
results = segment_medical_image(medical_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(results['gray'], cmap='gray')
axes[0, 0].set_title('原始医学图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(results['enhanced'], cmap='gray')
axes[0, 1].set_title('CLAHE增强', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(results['binary'], cmap='gray')
axes[0, 2].set_title(f"Otsu (T={results['threshold']})", fontsize=11)
axes[0, 2].axis('off')

axes[1, 0].imshow(results['cleaned'], cmap='gray')
axes[1, 0].set_title('形态学清理', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(results['result'], cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f"检测结果: {results['contours']} 个目标", fontsize=11)
axes[1, 1].axis('off')

# 直方图
axes[1, 2].hist(results['gray'].ravel(), 256, [0, 256], alpha=0.5, label='原始')
axes[1, 2].hist(results['enhanced'].ravel(), 256, [0, 256], alpha=0.5, label='增强后')
axes[1, 2].axvline(x=results['threshold'], color='r', linestyle='--',
                   label=f"Otsu T={results['threshold']}")
axes[1, 2].set_title('直方图', fontsize=11)
axes[1, 2].legend()

plt.suptitle('医学图像分割', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('medical_segmentation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"分割结果：检测到 {results['contours']} 个目标")
