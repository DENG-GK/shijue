# -*- coding: utf-8 -*-
"""
对不同噪声的去噪效果对比
"""

import cv2
import numpy as np


def add_gaussian_noise(image, sigma=25):
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    h, w = image.shape[:2]
    num = int(h * w * amount)
    for _ in range(num):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 255 if np.random.random() > 0.5 else 0
    return noisy


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

    # 测试高斯噪声
    print("=" * 60)
    print("高斯噪声 (σ=25) 去噪效果对比")
    print("=" * 60)
    gaussian_noisy = add_gaussian_noise(img, 25)
    print(f"{'方法':<25} {'PSNR (dB)':<10}")
    print("-" * 35)
    print(f"{'噪声图像':<25} {psnr(img, gaussian_noisy):<10.2f}")
    print(f"{'均值滤波 5×5':<25} {psnr(img, cv2.blur(gaussian_noisy, (5,5))):<10.2f}")
    print(f"{'高斯滤波 5×5':<25} {psnr(img, cv2.GaussianBlur(gaussian_noisy, (5,5), 0)):<10.2f}")
    print(f"{'中值滤波 5':<25} {psnr(img, cv2.medianBlur(gaussian_noisy, 5)):<10.2f}")
    print(f"{'双边滤波 (5,75,75)':<25} {psnr(img, cv2.bilateralFilter(gaussian_noisy, 5, 75, 75)):<10.2f}")

    # 测试椒盐噪声
    print("\n" + "=" * 60)
    print("椒盐噪声 (5%) 去噪效果对比")
    print("=" * 60)
    sp_noisy = add_salt_pepper_noise(img, 0.05)
    print(f"{'方法':<25} {'PSNR (dB)':<10}")
    print("-" * 35)
    print(f"{'噪声图像':<25} {psnr(img, sp_noisy):<10.2f}")
    print(f"{'均值滤波 5×5':<25} {psnr(img, cv2.blur(sp_noisy, (5,5))):<10.2f}")
    print(f"{'高斯滤波 5×5':<25} {psnr(img, cv2.GaussianBlur(sp_noisy, (5,5), 0)):<10.2f}")
    print(f"{'中值滤波 5':<25} {psnr(img, cv2.medianBlur(sp_noisy, 5)):<10.2f}")
    print(f"{'双边滤波 (5,75,75)':<25} {psnr(img, cv2.bilateralFilter(sp_noisy, 5, 75, 75)):<10.2f}")

    print("\n结论：")
    print("- 高斯噪声：高斯滤波效果最好")
    print("- 椒盐噪声：中值滤波效果最好（远超其他滤波器！）")


if __name__ == "__main__":
    main()
