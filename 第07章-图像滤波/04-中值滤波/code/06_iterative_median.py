# -*- coding: utf-8 -*-
"""
迭代中值滤波去除严重噪声
"""

import cv2
import numpy as np


def add_salt_pepper_noise(image, amount):
    noisy = image.copy()
    h, w = image.shape[:2]
    num = int(h * w * amount)
    for _ in range(num):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 255
    for _ in range(num):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 0
    return noisy


def psnr(original, processed):
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.ones((200, 200), dtype=np.uint8) * 128
        cv2.circle(img, (100, 100), 50, 200, -1)
        print("未找到测试图片，使用生成的示例图像")

    # 添加严重的椒盐噪声（20%）
    noisy = add_salt_pepper_noise(img, 0.20)

    # 单次大核 vs 多次小核
    single_large = cv2.medianBlur(noisy, 9)

    multi_small = noisy.copy()
    for _ in range(3):
        multi_small = cv2.medianBlur(multi_small, 5)

    # 显示对比
    cv2.imshow("Original", img)
    cv2.imshow("Heavy Noise (20%)", noisy)
    cv2.imshow("Single Median 9", single_large)
    cv2.imshow("3x Median 5", multi_small)

    # 计算 PSNR
    p_noisy = psnr(img, noisy)
    p_single = psnr(img, single_large)
    p_multi = psnr(img, multi_small)

    print("迭代中值滤波演示")
    print("=" * 40)
    print(f"噪声图像 PSNR：{p_noisy:.2f} dB")
    print(f"单次 Median 9：{p_single:.2f} dB")
    print(f"3次 Median 5： {p_multi:.2f} dB")
    print()
    print("结论：")
    print("- 单次大核：可能丢失细节")
    print("- 多次小核：逐步去噪，保留更多细节")
    print("- 对于严重噪声，可以先用小核多次处理")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
