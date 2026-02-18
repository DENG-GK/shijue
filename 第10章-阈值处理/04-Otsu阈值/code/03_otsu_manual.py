"""
手动实现Otsu算法
不使用OpenCV的THRESH_OTSU，从零实现并与OpenCV结果对比
"""

import cv2
import numpy as np

# ===================== 创建双峰图像 =====================

def create_bimodal_image():
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:, :200] = np.random.normal(60, 15, (300, 200)).clip(0, 255)
    img[:, 200:] = np.random.normal(190, 15, (300, 200)).clip(0, 255)
    return img.astype(np.uint8)

# ===================== 手动实现Otsu =====================

def otsu_threshold_manual(image):
    """
    手动实现Otsu阈值算法

    Parameters:
    -----------
    image : numpy.ndarray - 输入灰度图像

    Returns:
    --------
    optimal_threshold : int - 最佳阈值
    binary : numpy.ndarray - 二值化结果
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 计算直方图
    hist = np.zeros(256)
    for pixel in gray.ravel():
        hist[pixel] += 1

    # 归一化
    total_pixels = gray.size
    hist_norm = hist / total_pixels

    # 全局均值
    global_mean = np.sum(np.arange(256) * hist_norm)

    # 遍历所有可能的阈值
    max_variance = 0
    optimal_threshold = 0

    cumulative_sum = 0
    cumulative_mean = 0

    for T in range(256):
        cumulative_sum += hist_norm[T]
        cumulative_mean += T * hist_norm[T]

        if cumulative_sum == 0 or cumulative_sum == 1:
            continue

        w0 = cumulative_sum
        w1 = 1 - cumulative_sum

        mu0 = cumulative_mean / w0
        mu1 = (global_mean - cumulative_mean) / w1

        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > max_variance:
            max_variance = variance
            optimal_threshold = T

    # 应用阈值
    binary = np.where(gray > optimal_threshold, 255, 0).astype(np.uint8)

    return optimal_threshold, binary

# ===================== 对比测试 =====================

test_img = create_bimodal_image()

# 手动实现
manual_thresh, manual_binary = otsu_threshold_manual(test_img)

# OpenCV实现
cv_thresh, cv_binary = cv2.threshold(test_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("=" * 50)
print("Otsu算法实现对比")
print("=" * 50)
print(f"手动实现的Otsu阈值: {manual_thresh}")
print(f"OpenCV的Otsu阈值: {cv_thresh:.0f}")
print(f"两者是否一致: {manual_thresh == int(cv_thresh)}")
print("=" * 50)
