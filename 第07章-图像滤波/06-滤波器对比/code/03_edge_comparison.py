# -*- coding: utf-8 -*-
"""
边缘保持能力对比
"""

import cv2
import numpy as np


def main():
    # 创建有清晰边缘的测试图像
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 255  # 白色方块

    # 添加一些噪声
    noisy = img.copy()
    noise = np.random.normal(0, 10, img.shape)
    noisy = np.clip(noisy.astype(float) + noise, 0, 255).astype(np.uint8)

    # 滤波处理
    blur = cv2.blur(noisy, (5, 5))
    gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
    median = cv2.medianBlur(noisy, 5)
    bilateral = cv2.bilateralFilter(noisy, 5, 75, 75)

    # 边缘检测
    edges_orig = cv2.Canny(img, 50, 150)
    edges_blur = cv2.Canny(blur, 50, 150)
    edges_gauss = cv2.Canny(gaussian, 50, 150)
    edges_median = cv2.Canny(median, 50, 150)
    edges_bilateral = cv2.Canny(bilateral, 50, 150)

    # 显示滤波结果
    cv2.imshow("Original", img)
    cv2.imshow("Noisy", noisy)
    cv2.imshow("Blur", blur)
    cv2.imshow("Gaussian", gaussian)
    cv2.imshow("Median", median)
    cv2.imshow("Bilateral", bilateral)

    print("边缘保持能力对比")
    print("按任意键查看边缘检测结果...")
    cv2.waitKey(0)

    cv2.imshow("Edge - Original", edges_orig)
    cv2.imshow("Edge - Blur", edges_blur)
    cv2.imshow("Edge - Gaussian", edges_gauss)
    cv2.imshow("Edge - Median", edges_median)
    cv2.imshow("Edge - Bilateral", edges_bilateral)

    print("\n观察边缘检测结果：")
    print("- 均值滤波：边缘最模糊、断断续续")
    print("- 高斯滤波：边缘略模糊")
    print("- 中值滤波：边缘较清晰")
    print("- 双边滤波：边缘最清晰！")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
