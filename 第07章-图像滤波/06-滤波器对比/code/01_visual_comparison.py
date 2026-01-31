# -*- coding: utf-8 -*-
"""
四种滤波器视觉效果对比
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), (200, 200, 200), -1)
        cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)
        cv2.line(img, (0, 0), (300, 300), (0, 255, 0), 3)
        print("未找到测试图片，使用生成的示例图像")

    # 统一使用 5×5 核进行对比
    blur = cv2.blur(img, (5, 5))
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    median = cv2.medianBlur(img, 5)
    bilateral = cv2.bilateralFilter(img, 5, 75, 75)

    # 显示所有效果
    cv2.imshow("Original", img)
    cv2.imshow("Blur (Mean)", blur)
    cv2.imshow("Gaussian", gaussian)
    cv2.imshow("Median", median)
    cv2.imshow("Bilateral", bilateral)

    print("四种滤波器视觉效果对比")
    print("=" * 40)
    print("所有滤波器使用相同核大小 5×5")
    print()
    print("观察要点：")
    print("- 均值：边缘最模糊")
    print("- 高斯：比均值稍自然")
    print("- 中值：边缘保持较好")
    print("- 双边：边缘最锐利")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
