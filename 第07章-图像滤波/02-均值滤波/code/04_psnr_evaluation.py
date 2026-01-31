# -*- coding: utf-8 -*-
"""
均值滤波的去噪效果量化评估
使用 PSNR 指标评估不同核大小的效果
"""

import cv2
import numpy as np


def psnr(original, processed):
    """计算峰值信噪比 (PSNR)，值越大说明越接近原图"""
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def main():
    # 读取原始清晰图像
    img = cv2.imread("../images/test.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.ones((200, 200), dtype=np.uint8) * 128
        cv2.circle(img, (100, 100), 50, 200, -1)
        print("未找到测试图片，使用生成的示例图像\n")

    # 添加高斯噪声
    noisy = img.astype(np.float64) + np.random.normal(0, 25, img.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    print("均值滤波去噪效果量化评估")
    print("=" * 40)
    print(f"噪声图像 PSNR：{psnr(img, noisy):.2f} dB")
    print("-" * 40)

    # 不同核大小的去噪效果
    best_psnr = 0
    best_k = 0
    for k in [3, 5, 7, 9, 11, 15]:
        denoised = cv2.blur(noisy, (k, k))
        p = psnr(img, denoised)
        print(f"均值滤波 {k:2d}×{k:2d} → PSNR：{p:.2f} dB")
        if p > best_psnr:
            best_psnr = p
            best_k = k

    print("-" * 40)
    print(f"最佳核大小：{best_k}×{best_k}，PSNR：{best_psnr:.2f} dB")
    print("\n结论：核不是越大越好！存在一个最佳平衡点")


if __name__ == "__main__":
    main()
