"""
Canny边缘检测的自动阈值选择方法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 不同灰度的区域
    img[:150, :200] = 80
    img[:150, 200:] = 120
    img[150:, :200] = 160
    img[150:, 200:] = 200

    # 添加形状
    cv2.circle(img, (100, 75), 40, 200, -1)
    cv2.rectangle(img, (250, 30), (350, 130), 60, -1)
    cv2.circle(img, (100, 220), 40, 60, -1)
    cv2.rectangle(img, (250, 180), (350, 280), 220, -1)

    # 添加噪声
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

img = create_test_image()

# 高斯模糊
img_blur = cv2.GaussianBlur(img, (5, 5), 1.0)

print("测试图像已创建")

# ===================== 自动阈值方法 =====================

def auto_canny_median(img, sigma=0.33):
    """基于中值的自动阈值"""
    median = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return lower, upper

def auto_canny_otsu(img):
    """基于Otsu的自动阈值"""
    # 使用Otsu方法找到最佳阈值
    otsu_thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower = otsu_thresh * 0.5
    upper = otsu_thresh
    return int(lower), int(upper)

def auto_canny_percentile(img, low_percentile=10, high_percentile=90):
    """基于百分位数的自动阈值"""
    lower = np.percentile(img, low_percentile)
    upper = np.percentile(img, high_percentile)
    return int(lower), int(upper)

# 计算不同方法的阈值
thresh_fixed = (50, 150)
thresh_median = auto_canny_median(img_blur)
thresh_otsu = auto_canny_otsu(img_blur)
thresh_percentile = auto_canny_percentile(img_blur)

print(f"\n不同方法计算的阈值：")
print(f"  固定阈值:     {thresh_fixed}")
print(f"  中值方法:     {thresh_median}")
print(f"  Otsu方法:     {thresh_otsu}")
print(f"  百分位方法:   {thresh_percentile}")

# ===================== 应用不同阈值 =====================

edges_fixed = cv2.Canny(img_blur, *thresh_fixed)
edges_median = cv2.Canny(img_blur, *thresh_median)
edges_otsu = cv2.Canny(img_blur, *thresh_otsu)
edges_percentile = cv2.Canny(img_blur, *thresh_percentile)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=11)
axes[0, 0].axis('off')

# 直方图
axes[0, 1].hist(img.ravel(), bins=50, color='steelblue', alpha=0.7)
axes[0, 1].axvline(x=thresh_median[0], color='r', linestyle='--', label=f'低阈值')
axes[0, 1].axvline(x=thresh_median[1], color='g', linestyle='--', label=f'高阈值')
axes[0, 1].set_title('灰度直方图', fontsize=11)
axes[0, 1].set_xlabel('灰度值')
axes[0, 1].set_ylabel('像素数')
axes[0, 1].legend()

# 固定阈值
axes[0, 2].imshow(edges_fixed, cmap='gray')
axes[0, 2].set_title(f'固定阈值 {thresh_fixed}', fontsize=11)
axes[0, 2].axis('off')

# 中值方法
axes[1, 0].imshow(edges_median, cmap='gray')
axes[1, 0].set_title(f'中值方法 {thresh_median}\n(σ=0.33)', fontsize=11)
axes[1, 0].axis('off')

# Otsu方法
axes[1, 1].imshow(edges_otsu, cmap='gray')
axes[1, 1].set_title(f'Otsu方法 {thresh_otsu}', fontsize=11)
axes[1, 1].axis('off')

# 百分位方法
axes[1, 2].imshow(edges_percentile, cmap='gray')
axes[1, 2].set_title(f'百分位方法 {thresh_percentile}\n(10%, 90%)', fontsize=11)
axes[1, 2].axis('off')

plt.suptitle('Canny自动阈值选择方法', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('canny_auto_threshold.png', dpi=150, bbox_inches='tight')
plt.show()

# 封装成函数
def auto_canny(img, sigma=0.33):
    """
    自动Canny边缘检测
    使用基于中值的自动阈值
    """
    # 先进行高斯模糊
    blurred = cv2.GaussianBlur(img, (5, 5), 1.0)

    # 计算自动阈值
    median = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    # 应用Canny
    edges = cv2.Canny(blurred, lower, upper)

    return edges

print("\n自动Canny函数示例：")
print("  edges = auto_canny(img, sigma=0.33)")
print("\n图像已保存为 'canny_auto_threshold.png'")
