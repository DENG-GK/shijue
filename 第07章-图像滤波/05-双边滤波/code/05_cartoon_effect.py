# -*- coding: utf-8 -*-
"""
卡通风格化效果
使用双边滤波实现卡通渲染
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 200), (255, 100, 100), -1)
        cv2.rectangle(img, (150, 100), (250, 250), (100, 255, 100), -1)
        cv2.circle(img, (150, 150), 60, (100, 100, 255), -1)
        print("未找到测试图片，使用生成的示例图像")

    # 卡通化步骤：
    print("卡通风格化处理中...")

    # 1. 多次双边滤波使颜色平坦化
    cartoon = img.copy()
    for i in range(5):
        cartoon = cv2.bilateralFilter(cartoon, 9, 75, 75)
        print(f"  双边滤波迭代 {i+1}/5")

    # 2. 转灰度并用中值滤波去噪
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    # 3. 自适应阈值提取边缘
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )

    # 4. 将边缘与平坦化的彩色图合并
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon_final = cv2.bitwise_and(cartoon, edges_color)

    # 显示
    cv2.imshow("Original", img)
    cv2.imshow("Bilateral x5", cartoon)
    cv2.imshow("Edges", edges)
    cv2.imshow("Cartoon Effect", cartoon_final)

    print("\n卡通风格化步骤：")
    print("1. 多次双边滤波 → 颜色平坦化")
    print("2. 灰度 + 中值滤波 → 去噪")
    print("3. 自适应阈值 → 提取边缘")
    print("4. 颜色图 AND 边缘 → 卡通效果")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
