# -*- coding: utf-8 -*-
"""
滤波器选择器（交互式工具）
使用滑动条实时对比不同滤波器效果
"""

import cv2
import numpy as np


def nothing(x):
    pass


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        img = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), (200, 200, 200), -1)
        cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)
        print("未找到测试图片，使用生成的示例图像")

    # 创建窗口
    cv2.namedWindow("Filter Comparison")

    # 创建滑动条
    cv2.createTrackbar("Filter", "Filter Comparison", 0, 3, nothing)
    # 0=均值, 1=高斯, 2=中值, 3=双边
    cv2.createTrackbar("Kernel", "Filter Comparison", 2, 10, nothing)
    # 核大小 = 2 * value + 1（确保是奇数）

    filter_names = ["Mean Blur", "Gaussian", "Median", "Bilateral"]

    print("滤波器交互选择器")
    print("=" * 40)
    print("Filter 滑动条：")
    print("  0 = 均值滤波")
    print("  1 = 高斯滤波")
    print("  2 = 中值滤波")
    print("  3 = 双边滤波")
    print()
    print("Kernel 滑动条：控制核大小（自动转为奇数）")
    print()
    print("按 ESC 退出")

    while True:
        filter_type = cv2.getTrackbarPos("Filter", "Filter Comparison")
        kernel_val = cv2.getTrackbarPos("Kernel", "Filter Comparison")
        kernel_size = 2 * kernel_val + 1  # 确保是奇数
        if kernel_size < 3:
            kernel_size = 3

        # 根据选择应用滤波
        if filter_type == 0:
            result = cv2.blur(img, (kernel_size, kernel_size))
        elif filter_type == 1:
            result = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif filter_type == 2:
            result = cv2.medianBlur(img, kernel_size)
        else:
            result = cv2.bilateralFilter(img, kernel_size, 75, 75)

        # 显示信息
        display = result.copy()
        text = f"{filter_names[filter_type]} - Kernel: {kernel_size}x{kernel_size}"
        cv2.putText(display, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Filter Comparison", display)

        key = cv2.waitKey(30)
        if key == 27:  # ESC 退出
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
