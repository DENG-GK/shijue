# -*- coding: utf-8 -*-
"""
边界填充效果可视化
演示 OpenCV 中不同的边界填充方式
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        # 创建示例图像
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        img[25:125, 25:125] = [255, 200, 100]
        cv2.circle(img, (75, 75), 30, (0, 0, 255), -1)
        print("未找到测试图片，使用生成的示例图像")

    # 填充大小（像素）
    top = bottom = left = right = 50

    # 不同的边界填充方式
    constant = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    replicate = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_REPLICATE
    )

    reflect = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_REFLECT
    )

    reflect101 = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_REFLECT_101
    )

    wrap = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_WRAP
    )

    # 显示所有填充效果
    cv2.imshow("Original", img)
    cv2.imshow("CONSTANT (Black)", constant)
    cv2.imshow("REPLICATE", replicate)
    cv2.imshow("REFLECT", reflect)
    cv2.imshow("REFLECT_101 (Default)", reflect101)
    cv2.imshow("WRAP", wrap)

    print("边界填充方式说明：")
    print("1. CONSTANT - 用固定值填充（这里是黑色）")
    print("2. REPLICATE - 复制边缘像素")
    print("3. REFLECT - 镜像反射")
    print("4. REFLECT_101 - 不含边界的反射（OpenCV 默认）")
    print("5. WRAP - 环绕（左右/上下连接）")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
