# -*- coding: utf-8 -*-
"""
基础双边滤波演示
展示不同参数强度的双边滤波效果
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
        # 添加一些纹理
        for i in range(50, 250, 10):
            cv2.line(img, (i, 50), (i, 250), (180, 180, 180), 1)
        print("未找到测试图片，使用生成的示例图像")

    # ========== cv2.bilateralFilter() 基本用法 ==========
    # 参数：(图像, 邻域直径d, sigmaColor, sigmaSpace)

    # 轻度双边滤波
    bilateral_light = cv2.bilateralFilter(img, 5, 50, 50)

    # 中度双边滤波（常用参数）
    bilateral_medium = cv2.bilateralFilter(img, 9, 75, 75)

    # 强度双边滤波
    bilateral_strong = cv2.bilateralFilter(img, 15, 100, 100)

    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Bilateral Light (5, 50, 50)", bilateral_light)
    cv2.imshow("Bilateral Medium (9, 75, 75)", bilateral_medium)
    cv2.imshow("Bilateral Strong (15, 100, 100)", bilateral_strong)

    print("双边滤波基础演示")
    print("=" * 40)
    print("参数说明：")
    print("- d：邻域直径（越大越慢）")
    print("- sigmaColor：颜色差异容忍度")
    print("- sigmaSpace：空间距离影响范围")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
