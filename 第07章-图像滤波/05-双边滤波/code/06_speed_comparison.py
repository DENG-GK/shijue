# -*- coding: utf-8 -*-
"""
双边滤波的速度对比
展示双边滤波相比其他滤波器的性能差异
"""

import cv2
import numpy as np
import time


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        print("未找到测试图片，使用随机生成的图像")

    # 测试不同滤波器的速度
    iterations = 10

    print("滤波器速度对比测试")
    print("=" * 45)
    print(f"图像大小：{img.shape[1]}×{img.shape[0]}")
    print(f"测试次数：{iterations}")
    print(f"核大小：15×15 (双边滤波 d=15)")
    print("-" * 45)

    # 均值滤波
    start = time.time()
    for _ in range(iterations):
        cv2.blur(img, (15, 15))
    blur_time = (time.time() - start) / iterations * 1000
    print(f"均值滤波：    {blur_time:.2f} ms")

    # 高斯滤波
    start = time.time()
    for _ in range(iterations):
        cv2.GaussianBlur(img, (15, 15), 0)
    gaussian_time = (time.time() - start) / iterations * 1000
    print(f"高斯滤波：    {gaussian_time:.2f} ms")

    # 中值滤波
    start = time.time()
    for _ in range(iterations):
        cv2.medianBlur(img, 15)
    median_time = (time.time() - start) / iterations * 1000
    print(f"中值滤波：    {median_time:.2f} ms")

    # 双边滤波（最慢！）
    start = time.time()
    for _ in range(iterations):
        cv2.bilateralFilter(img, 15, 75, 75)
    bilateral_time = (time.time() - start) / iterations * 1000
    print(f"双边滤波：    {bilateral_time:.2f} ms")

    print("-" * 45)
    print(f"双边滤波比均值滤波慢 {bilateral_time/blur_time:.1f} 倍")
    print(f"双边滤波比高斯滤波慢 {bilateral_time/gaussian_time:.1f} 倍")
    print()
    print("结论：双边滤波的保边效果是以速度为代价的！")
    print("实时应用需要考虑使用 GPU 加速或其他优化方案")


if __name__ == "__main__":
    main()
