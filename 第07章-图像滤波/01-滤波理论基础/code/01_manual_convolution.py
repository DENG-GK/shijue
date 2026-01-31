# -*- coding: utf-8 -*-
"""
手动实现卷积操作
通过手动实现，深入理解卷积的工作原理
"""

import cv2
import numpy as np


def manual_convolution(image, kernel):
    """
    手动实现2D卷积操作

    参数：
        image: 输入灰度图像 (H, W)
        kernel: 卷积核 (kH, kW)

    返回：
        卷积结果图像
    """
    # 获取图像和卷积核的尺寸
    img_h, img_w = image.shape[:2]
    ker_h, ker_w = kernel.shape[:2]

    # 计算padding大小（保证输出图像和输入大小一样）
    pad_h = ker_h // 2
    pad_w = ker_w // 2

    # 对图像进行边界填充（用镜像反射）
    padded = cv2.copyMakeBorder(
        image, pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_REFLECT_101
    )

    # 创建输出图像
    output = np.zeros_like(image, dtype=np.float64)

    # 逐像素计算卷积
    for i in range(img_h):
        for j in range(img_w):
            # 取出当前位置对应的邻域
            region = padded[i:i+ker_h, j:j+ker_w].astype(np.float64)
            # 逐元素相乘后求和
            output[i, j] = np.sum(region * kernel)

    # 裁剪到 0-255 并转为 uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def main():
    # 读取图像并转为灰度
    img = cv2.imread("../images/test.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 如果没有测试图片，创建一个示例图像
        img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        print("未找到测试图片，使用随机生成的图像")

    # 定义一个 3×3 均值卷积核
    kernel_mean = np.ones((3, 3), dtype=np.float64) / 9.0
    print("均值卷积核：")
    print(kernel_mean)

    # 手动卷积
    result_manual = manual_convolution(img, kernel_mean)

    # OpenCV 内置卷积（对比用）
    result_opencv = cv2.filter2D(img, -1, kernel_mean)

    # 对比结果（应该非常接近）
    diff = cv2.absdiff(result_manual, result_opencv)
    print(f"\n手动实现与 OpenCV 的最大差异：{diff.max()}")
    # 通常差异为 0 或 1（浮点精度导致）

    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Manual Convolution", result_manual)
    cv2.imshow("OpenCV filter2D", result_opencv)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
