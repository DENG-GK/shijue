# -*- coding: utf-8 -*-
"""
基础高斯滤波演示
展示不同核大小的高斯滤波效果
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

    # ========== 不同核大小的高斯滤波（σ 自动计算）==========
    gauss_3 = cv2.GaussianBlur(img, (3, 3), 0)
    gauss_5 = cv2.GaussianBlur(img, (5, 5), 0)
    gauss_9 = cv2.GaussianBlur(img, (9, 9), 0)
    gauss_15 = cv2.GaussianBlur(img, (15, 15), 0)

    # 显示对比
    cv2.imshow("Original", img)
    cv2.imshow("Gaussian 3x3", gauss_3)
    cv2.imshow("Gaussian 5x5", gauss_5)
    cv2.imshow("Gaussian 9x9", gauss_9)
    cv2.imshow("Gaussian 15x15", gauss_15)

    print("高斯滤波效果对比：")
    print("核越大，模糊效果越强，但比均值滤波更自然")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
