# -*- coding: utf-8 -*-
"""
boxFilter 的 normalize 参数演示
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        print("未找到测试图片，使用生成的示例图像")

    # normalize=True → 均值滤波（和 blur 一样）
    box_norm = cv2.boxFilter(img, -1, (5, 5), normalize=True)

    # normalize=False → 求和滤波（不除以核面积）
    # 注意：结果可能非常大，uint8 会溢出截断到 255
    box_sum = cv2.boxFilter(img, -1, (5, 5), normalize=False)

    # 使用更大深度避免溢出
    box_sum_float = cv2.boxFilter(img, cv2.CV_64F, (5, 5), normalize=False)

    print(f"原图最大值：{img.max()}")
    print(f"归一化后最大值：{box_norm.max()}")
    print(f"未归一化（截断后）最大值：{box_sum.max()}")
    print(f"未归一化（浮点）最大值：{box_sum_float.max():.2f}")

    # 显示
    cv2.imshow("Original", img)
    cv2.imshow("boxFilter normalize=True", box_norm)
    cv2.imshow("boxFilter normalize=False (clipped)", box_sum)

    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
