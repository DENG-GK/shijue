# -*- coding: utf-8 -*-
"""
滤波器速度对比
"""

import cv2
import numpy as np
import time


def main():
    # 创建测试图像
    img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)

    # 测试参数
    iterations = 50
    kernel_sizes = [3, 5, 9, 15]

    print("滤波器速度对比")
    print("图像大小：500×500")
    print("测试次数：50 次取平均")
    print("=" * 70)
    print(f"{'核大小':<10} {'均值(ms)':<12} {'高斯(ms)':<12} {'中值(ms)':<12} {'双边(ms)':<12}")
    print("-" * 70)

    for k in kernel_sizes:
        # 均值
        start = time.time()
        for _ in range(iterations):
            cv2.blur(img, (k, k))
        blur_time = (time.time() - start) / iterations * 1000

        # 高斯
        start = time.time()
        for _ in range(iterations):
            cv2.GaussianBlur(img, (k, k), 0)
        gauss_time = (time.time() - start) / iterations * 1000

        # 中值
        start = time.time()
        for _ in range(iterations):
            cv2.medianBlur(img, k)
        median_time = (time.time() - start) / iterations * 1000

        # 双边
        start = time.time()
        for _ in range(iterations):
            cv2.bilateralFilter(img, k, 75, 75)
        bilateral_time = (time.time() - start) / iterations * 1000

        print(f"{k}×{k:<8} {blur_time:<12.2f} {gauss_time:<12.2f} {median_time:<12.2f} {bilateral_time:<12.2f}")

    print("-" * 70)
    print("\n结论：")
    print("- 均值滤波最快")
    print("- 高斯滤波也很快")
    print("- 中值滤波中等速度")
    print("- 双边滤波最慢（慢几十倍！）")


if __name__ == "__main__":
    main()
