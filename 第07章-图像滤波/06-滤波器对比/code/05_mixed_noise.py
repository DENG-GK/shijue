# -*- coding: utf-8 -*-
"""
混合噪声的两步去噪
"""

import cv2
import numpy as np


def add_mixed_noise(image, gaussian_sigma=15, sp_amount=0.02):
    """添加混合噪声：高斯 + 椒盐"""
    # 先加高斯噪声
    noisy = image.astype(float) + np.random.normal(0, gaussian_sigma, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # 再加椒盐噪声
    h, w = noisy.shape[:2]
    num = int(h * w * sp_amount)
    for _ in range(num):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 255 if np.random.random() > 0.5 else 0
    return noisy


def psnr(original, processed):
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.ones((200, 200), dtype=np.uint8) * 128
        cv2.circle(img, (100, 100), 50, 200, -1)
        print("未找到测试图片，使用生成的示例图像\n")

    # 添加混合噪声
    noisy = add_mixed_noise(img, gaussian_sigma=20, sp_amount=0.03)

    # 单一滤波器处理
    only_gauss = cv2.GaussianBlur(noisy, (5, 5), 0)
    only_median = cv2.medianBlur(noisy, 5)

    # 两步去噪：先中值去椒盐，再高斯去高斯噪声
    step1 = cv2.medianBlur(noisy, 3)  # 先用小核中值去椒盐
    step2 = cv2.GaussianBlur(step1, (5, 5), 0)  # 再高斯去剩余噪声

    print("混合噪声（高斯+椒盐）去噪对比")
    print("=" * 45)
    print(f"{'方法':<30} {'PSNR (dB)':<10}")
    print("-" * 45)
    print(f"{'噪声图像':<30} {psnr(img, noisy):<10.2f}")
    print(f"{'只用高斯滤波':<30} {psnr(img, only_gauss):<10.2f}")
    print(f"{'只用中值滤波':<30} {psnr(img, only_median):<10.2f}")
    print(f"{'两步：先中值后高斯':<30} {psnr(img, step2):<10.2f}")
    print("-" * 45)
    print("\n结论：对混合噪声，两步去噪效果最好！")
    print("策略：先用中值滤波去除椒盐噪声，再用高斯滤波去除高斯噪声")

    # 显示
    cv2.imshow("Original", img)
    cv2.imshow("Mixed Noise", noisy)
    cv2.imshow("Only Gaussian", only_gauss)
    cv2.imshow("Only Median", only_median)
    cv2.imshow("Two-Step: Median + Gaussian", step2)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
