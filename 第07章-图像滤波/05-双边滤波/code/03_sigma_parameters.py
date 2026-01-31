# -*- coding: utf-8 -*-
"""
sigmaColor 和 sigmaSpace 的影响
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:, 150:] = 200
        cv2.circle(img, (150, 150), 80, (0, 0, 255), -1)
        print("未找到测试图片，使用生成的示例图像")

    print("sigmaColor 的影响（固定 sigmaSpace=75）")
    print("=" * 40)

    # sigmaColor 的影响（固定 sigmaSpace=75）
    colors = [25, 50, 75, 100, 150]
    for sc in colors:
        result = cv2.bilateralFilter(img, 9, sc, 75)
        cv2.imshow(f"sigmaColor={sc}, sigmaSpace=75", result)

    cv2.imshow("Original", img)
    print("sigmaColor 越大，颜色差异大的像素也被混合，边缘开始模糊")
    print("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nsigmaSpace 的影响（固定 sigmaColor=75）")
    print("=" * 40)

    # sigmaSpace 的影响（固定 sigmaColor=75）
    spaces = [25, 50, 75, 100, 150]
    for ss in spaces:
        result = cv2.bilateralFilter(img, 9, 75, ss)
        cv2.imshow(f"sigmaColor=75, sigmaSpace={ss}", result)

    cv2.imshow("Original", img)
    print("sigmaSpace 越大，更远的像素参与计算，整体更平滑")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
