# -*- coding: utf-8 -*-
"""
查看高斯核的实际数值
"""

import cv2
import numpy as np


def main():
    print("高斯核的实际数值")
    print("=" * 50)

    # OpenCV 可以直接生成高斯核，方便我们查看
    kernel_3 = cv2.getGaussianKernel(3, 0)   # 3×1 的一维核，σ自动
    kernel_5 = cv2.getGaussianKernel(5, 0)   # 5×1 的一维核
    kernel_5_s1 = cv2.getGaussianKernel(5, 1.0)  # σ=1.0

    print("3×1 高斯核 (σ=auto)：")
    print(np.round(kernel_3.T, 4))
    # [[0.25  0.5   0.25]]

    print("\n5×1 高斯核 (σ=auto)：")
    print(np.round(kernel_5.T, 4))
    # [[0.0625  0.25  0.375  0.25  0.0625]]

    print("\n5×1 高斯核 (σ=1.0)：")
    print(np.round(kernel_5_s1.T, 4))
    # [[0.0545  0.2442  0.4026  0.2442  0.0545]]

    # 二维高斯核 = 一维核的外积
    kernel_2d = kernel_5 @ kernel_5.T
    print("\n5×5 二维高斯核：")
    print(np.round(kernel_2d, 4))
    # 中心值最大，四角值最小

    print(f"\n核的总和 = {kernel_2d.sum():.6f}")  # 应该接近 1.0

    print("\n观察：")
    print("- 中心权重最大")
    print("- 越靠近边缘，权重越小")
    print("- 所有权重之和 = 1（归一化）")


if __name__ == "__main__":
    main()
