# -*- coding: utf-8 -*-
"""
非方形核的均值滤波
实现方向性模糊效果
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (150, 150), 80, (255, 255, 255), -1)
        cv2.circle(img, (150, 150), 40, (0, 0, 255), -1)
        print("未找到测试图片，使用生成的示例图像")

    # 核不一定是正方形的！可以是长方形

    # 水平方向模糊（宽核）：水平方向平滑，垂直方向保留
    blur_h = cv2.blur(img, (15, 1))

    # 垂直方向模糊（高核）：垂直方向平滑，水平方向保留
    blur_v = cv2.blur(img, (1, 15))

    # 正方形模糊（对比用）
    blur_sq = cv2.blur(img, (15, 15))

    # 显示
    cv2.imshow("Original", img)
    cv2.imshow("Horizontal Blur (15,1)", blur_h)
    cv2.imshow("Vertical Blur (1,15)", blur_v)
    cv2.imshow("Square Blur (15,15)", blur_sq)

    print("方向性模糊效果：")
    print("- 水平核 (15,1)：模拟水平运动模糊")
    print("- 垂直核 (1,15)：模拟垂直运动模糊/雨滴效果")
    print("- 正方形核 (15,15)：均匀模糊")
    print("\n这在创意图像处理中很有用！")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
