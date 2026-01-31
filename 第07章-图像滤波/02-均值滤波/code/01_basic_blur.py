# -*- coding: utf-8 -*-
"""
基础均值滤波演示
展示不同核大小的均值滤波效果
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        # 创建示例图像
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), -1)
        cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)
        print("未找到测试图片，使用生成的示例图像")

    # ========== cv2.blur() 基本用法 ==========

    # 3×3 均值滤波
    blur_3 = cv2.blur(img, (3, 3))

    # 5×5 均值滤波
    blur_5 = cv2.blur(img, (5, 5))

    # 9×9 均值滤波
    blur_9 = cv2.blur(img, (9, 9))

    # 15×15 均值滤波（模糊效果很明显）
    blur_15 = cv2.blur(img, (15, 15))

    # 显示结果对比
    cv2.imshow("Original", img)
    cv2.imshow("Blur 3x3", blur_3)
    cv2.imshow("Blur 5x5", blur_5)
    cv2.imshow("Blur 9x9", blur_9)
    cv2.imshow("Blur 15x15", blur_15)

    print("均值滤波效果对比：")
    print("核越大，模糊效果越强")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
