# -*- coding: utf-8 -*-
"""
高斯滤波 vs 均值滤波对比
使用 PSNR 量化比较
"""

import cv2
import numpy as np


def add_gaussian_noise(image, sigma=25):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float64)
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255)
    return noisy.astype(np.uint8)


def psnr(original, processed):
    """计算 PSNR"""
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.ones((200, 200), dtype=np.uint8) * 128
        cv2.circle(img, (100, 100), 50, 200, -1)
        print("未找到测试图片，使用生成的示例图像\n")

    # 添加噪声
    noisy = add_gaussian_noise(img, sigma=30)

    print("高斯滤波 vs 均值滤波 PSNR 对比")
    print("噪声类型：高斯噪声 (σ=30)")
    print("=" * 50)
    print(f"{'方法':<20} {'PSNR (dB)':>12}")
    print("-" * 50)
    print(f"{'噪声图像':<20} {psnr(img, noisy):>12.2f}")

    # 不同核大小的对比
    for k in [3, 5, 7, 9, 11]:
        mean_result = cv2.blur(noisy, (k, k))
        gauss_result = cv2.GaussianBlur(noisy, (k, k), 0)

        p_mean = psnr(img, mean_result)
        p_gauss = psnr(img, gauss_result)

        print(f"均值 {k}×{k:<15} {p_mean:>12.2f}")
        print(f"高斯 {k}×{k:<15} {p_gauss:>12.2f}")
        print("-" * 50)

    print("\n结论：高斯滤波的 PSNR 通常略优于同大小的均值滤波")


if __name__ == "__main__":
    main()
