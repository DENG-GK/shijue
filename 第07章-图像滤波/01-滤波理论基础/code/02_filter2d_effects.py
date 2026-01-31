# -*- coding: utf-8 -*-
"""
OpenCV 的 filter2D 函数演示
展示不同卷积核的效果：模糊、锐化、边缘检测、浮雕
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        # 创建一个示例图像
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), -1)
        cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)
        print("未找到测试图片，使用生成的示例图像")

    # ========== 不同的卷积核 ==========

    # 1. 均值模糊核
    kernel_blur = np.ones((5, 5), dtype=np.float32) / 25
    result_blur = cv2.filter2D(img, -1, kernel_blur)

    # 2. 锐化核
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    result_sharpen = cv2.filter2D(img, -1, kernel_sharpen)

    # 3. 边缘检测核
    kernel_edge = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    result_edge = cv2.filter2D(img, -1, kernel_edge)

    # 4. 浮雕效果核
    kernel_emboss = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)
    result_emboss = cv2.filter2D(img, -1, kernel_emboss)

    # 显示所有效果
    cv2.imshow("Original", img)
    cv2.imshow("Blur (Mean)", result_blur)
    cv2.imshow("Sharpen", result_sharpen)
    cv2.imshow("Edge Detection", result_edge)
    cv2.imshow("Emboss", result_emboss)

    print("不同卷积核效果对比：")
    print("1. 均值模糊 - 图像变模糊")
    print("2. 锐化 - 边缘更清晰")
    print("3. 边缘检测 - 只保留边缘")
    print("4. 浮雕 - 立体浮雕效果")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
