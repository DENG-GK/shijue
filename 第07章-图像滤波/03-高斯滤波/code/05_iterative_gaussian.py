# -*- coding: utf-8 -*-
"""
多次迭代高斯滤波
小核多次 vs 大核一次
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), -1)
        cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)
        print("未找到测试图片，使用生成的示例图像")

    # 单次大核
    single_large = cv2.GaussianBlur(img, (15, 15), 0)

    # 多次小核
    multi_small = img.copy()
    for _ in range(5):
        multi_small = cv2.GaussianBlur(multi_small, (5, 5), 0)

    # 显示对比
    cv2.imshow("Original", img)
    cv2.imshow("Single (15,15)", single_large)
    cv2.imshow("5x GaussianBlur (5,5)", multi_small)

    print("多次迭代高斯滤波演示")
    print("=" * 40)
    print("单次大核 (15,15) vs 5次小核 (5,5)")
    print()
    print("数学性质：")
    print("多次高斯滤波等价于一次更大的高斯滤波")
    print("σ_total = √(σ1² + σ2² + ... + σn²)")
    print("如果每次 σ=1，做 n 次，等效 σ = √n")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
