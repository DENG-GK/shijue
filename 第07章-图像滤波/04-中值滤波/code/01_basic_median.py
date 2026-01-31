# -*- coding: utf-8 -*-
"""
基础中值滤波演示
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

    # 不同核大小的中值滤波
    # 注意：参数是单个数字，不是元组！
    median_3 = cv2.medianBlur(img, 3)
    median_5 = cv2.medianBlur(img, 5)
    median_9 = cv2.medianBlur(img, 9)
    median_15 = cv2.medianBlur(img, 15)

    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Median 3", median_3)
    cv2.imshow("Median 5", median_5)
    cv2.imshow("Median 9", median_9)
    cv2.imshow("Median 15", median_15)

    print("中值滤波基础演示")
    print("=" * 40)
    print("注意：cv2.medianBlur 的核大小参数是单个数字，不是元组！")
    print("- cv2.blur(img, (5, 5))       ← 元组")
    print("- cv2.medianBlur(img, 5)      ← 单个数字")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
