# -*- coding: utf-8 -*-
"""
中值滤波对高斯噪声的效果（不太好）
对比中值滤波和高斯滤波对高斯噪声的处理
"""

import cv2
import numpy as np


def add_gaussian_noise(image, sigma):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(float) + noise, 0, 255)
    return noisy.astype(np.uint8)


def psnr(original, processed):
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def main():
    # 读取灰度图像
    img = cv2.imread("../images/test.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.ones((200, 200), dtype=np.uint8) * 128
        cv2.circle(img, (100, 100), 50, 200, -1)
        print("未找到测试图片，使用生成的示例图像\n")

    noisy = add_gaussian_noise(img, sigma=25)

    print("高斯滤波 vs 中值滤波（对高斯噪声）")
    print("噪声类型：高斯噪声 (σ=25)")
    print("=" * 40)
    print(f"{'方法':<20} {'PSNR (dB)':<12}")
    print("-" * 40)
    print(f"{'噪声图像':<20} {psnr(img, noisy):<12.2f}")

    for k in [3, 5, 7, 9]:
        gauss = cv2.GaussianBlur(noisy, (k, k), 0)
        median = cv2.medianBlur(noisy, k)

        print(f"高斯 {k}×{k:<15} {psnr(img, gauss):<12.2f}")
        print(f"中值 {k:<18} {psnr(img, median):<12.2f}")

    print("-" * 40)
    print("\n结论：对高斯噪声，高斯滤波略优于中值滤波")
    print("不同噪声要用不同的滤波器！")


if __name__ == "__main__":
    main()
