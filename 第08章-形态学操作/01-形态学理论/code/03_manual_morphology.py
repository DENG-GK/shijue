"""
示例3：手动实现简化版腐蚀操作，帮助理解原理
注意：实际使用时请用 OpenCV 的内置函数，这里只是为了学习原理
"""

import cv2
import numpy as np

def manual_erosion(image, kernel):
    """
    手动实现腐蚀操作（简化版，仅用于教学）

    腐蚀的规则：
    只有当结构元素完全被图像前景覆盖时，输出才为前景（1）
    """
    # 获取图像和核的尺寸
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    # 计算padding（核心点到边缘的距离）
    pad_h, pad_w = k_h // 2, k_w // 2

    # 创建输出图像
    output = np.zeros_like(image)

    # 遍历每个像素
    for i in range(pad_h, img_h - pad_h):
        for j in range(pad_w, img_w - pad_w):
            # 提取当前窗口
            window = image[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]

            # 检查结构元素覆盖的位置
            # 只有所有位置都是前景（255）时，中心才保持前景
            check = window[kernel == 1]

            if np.all(check == 255):
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output

def manual_dilation(image, kernel):
    """
    手动实现膨胀操作（简化版，仅用于教学）

    膨胀的规则：
    只要结构元素与图像前景有任何重叠，输出就为前景（1）
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    output = np.zeros_like(image)

    for i in range(pad_h, img_h - pad_h):
        for j in range(pad_w, img_w - pad_w):
            window = image[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
            check = window[kernel == 1]

            # 只要有任何一个位置是前景，中心就变成前景
            if np.any(check == 255):
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output

# ===================== 测试手动实现 =====================

# 创建简单的测试图像
test = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 255, 255, 255, 0, 0],
    [0, 0, 255, 255, 255, 0, 0],
    [0, 0, 255, 255, 255, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
], dtype=np.uint8)

# 3×3 矩形核
kernel = np.ones((3, 3), dtype=np.uint8)

# 手动腐蚀
manual_eroded = manual_erosion(test, kernel)

# OpenCV 腐蚀（用于对比）
cv_eroded = cv2.erode(test, kernel, iterations=1)

print("原始图像:")
print((test / 255).astype(int))
print("\n手动腐蚀结果:")
print((manual_eroded / 255).astype(int))
print("\nOpenCV腐蚀结果:")
print((cv_eroded / 255).astype(int))
print("\n两种方法结果是否一致:", np.array_equal(manual_eroded, cv_eroded))
