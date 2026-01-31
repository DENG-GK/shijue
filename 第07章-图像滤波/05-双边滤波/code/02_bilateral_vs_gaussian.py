# -*- coding: utf-8 -*-
"""
双边滤波 vs 高斯滤波（边缘保持对比）
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        # 创建有清晰边缘的测试图像
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:, 150:] = 255
        cv2.circle(img, (150, 150), 80, (0, 0, 255), -1)
        print("未找到测试图片，使用生成的示例图像")

    # 高斯滤波（会模糊边缘）
    gaussian = cv2.GaussianBlur(img, (15, 15), 0)

    # 双边滤波（保持边缘）
    bilateral = cv2.bilateralFilter(img, 15, 75, 75)

    # 边缘检测对比
    edges_original = cv2.Canny(img, 100, 200)
    edges_gaussian = cv2.Canny(gaussian, 100, 200)
    edges_bilateral = cv2.Canny(bilateral, 100, 200)

    # 显示滤波结果
    cv2.imshow("Original", img)
    cv2.imshow("Gaussian 15x15", gaussian)
    cv2.imshow("Bilateral (15, 75, 75)", bilateral)

    print("滤波效果对比")
    print("按任意键查看边缘检测结果...")
    cv2.waitKey(0)

    cv2.imshow("Edges - Original", edges_original)
    cv2.imshow("Edges - Gaussian", edges_gaussian)
    cv2.imshow("Edges - Bilateral", edges_bilateral)

    print("\n边缘保持对比：")
    print("- 高斯滤波后的边缘变得模糊、断断续续")
    print("- 双边滤波后的边缘仍然清晰！")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
