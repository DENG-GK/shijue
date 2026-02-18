"""
文档扫描二值化增强
对比Otsu、自适应和组合方法对不均匀光照文档的处理效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 文档增强函数 =====================

def enhance_document(image, method='otsu'):
    """
    增强扫描文档的可读性

    Parameters:
    -----------
    image : numpy.ndarray - 输入图像
    method : str - 'otsu', 'adaptive', 'combined'
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 去噪
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    if method == 'otsu':
        _, enhanced = cv2.threshold(denoised, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == 'adaptive':
        enhanced = cv2.adaptiveThreshold(denoised, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 21, 10)

    elif method == 'combined':
        # 先用Otsu获取全局阈值
        otsu_thresh, _ = cv2.threshold(denoised, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 使用自适应阈值
        adaptive = cv2.adaptiveThreshold(denoised, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 21, 10)
        # 结合两者
        _, global_binary = cv2.threshold(denoised, otsu_thresh * 0.9, 255,
                                        cv2.THRESH_BINARY)
        enhanced = cv2.bitwise_and(adaptive, global_binary)

    else:
        raise ValueError(f"未知方法: {method}")

    return enhanced

# ===================== 创建模拟文档图像 =====================

def create_document_image():
    img = np.ones((500, 700), dtype=np.uint8) * 230  # 浅色纸张

    # 添加不均匀光照
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            shade = 1.0 - 0.25 * (i / rows) - 0.15 * np.sin(j / cols * np.pi)
            img[i, j] = int(img[i, j] * shade)

    # 添加模拟文字行
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        "Document Processing with OpenCV",
        "===============================",
        "",
        "This is a sample document that",
        "demonstrates thresholding techniques",
        "for document image enhancement.",
        "",
        "Key benefits:",
        "- Improved readability",
        "- Better OCR accuracy",
        "- Reduced file size",
    ]

    y = 50
    for line in lines:
        cv2.putText(img, line, (50, y), font, 0.6, 30, 1)
        y += 35

    # 添加噪声
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# ===================== 处理文档 =====================

doc_img = create_document_image()

result_otsu = enhance_document(doc_img, 'otsu')
result_adaptive = enhance_document(doc_img, 'adaptive')
result_combined = enhance_document(doc_img, 'combined')

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].imshow(doc_img, cmap='gray')
axes[0, 0].set_title('原始文档\n(光照不均匀)', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(result_otsu, cmap='gray')
axes[0, 1].set_title('Otsu阈值\n(全局阈值)', fontsize=12)
axes[0, 1].axis('off')

axes[1, 0].imshow(result_adaptive, cmap='gray')
axes[1, 0].set_title('自适应阈值\n(局部阈值)', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(result_combined, cmap='gray')
axes[1, 1].set_title('组合方法\n(Otsu + 自适应)', fontsize=12)
axes[1, 1].axis('off')

plt.suptitle('文档扫描二值化增强', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('document_enhancement.png', dpi=150, bbox_inches='tight')
plt.show()

print("文档增强方法对比：")
print("- Otsu: 快速，但在光照不均时效果差")
print("- Adaptive: 处理光照不均效果好，但可能引入噪声")
print("- Combined: 结合两者优点，通常效果最好")
