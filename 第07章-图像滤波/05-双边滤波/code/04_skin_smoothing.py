# -*- coding: utf-8 -*-
"""
人像磨皮效果演示
"""

import cv2
import numpy as np


def main():
    # 读取人像图片
    img = cv2.imread("../images/portrait.jpg")
    if img is None:
        # 创建模拟的"人像"
        img = np.ones((300, 300, 3), dtype=np.uint8) * 200
        # 添加一些模拟的皮肤瑕疵
        for _ in range(100):
            y, x = np.random.randint(50, 250), np.random.randint(50, 250)
            color = np.random.randint(150, 220)
            cv2.circle(img, (x, y), np.random.randint(1, 4),
                       (color, color, color), -1)
        print("未找到人像图片，使用模拟图像演示")

    # 方法1：高斯滤波磨皮（边缘会模糊）
    gaussian_skin = cv2.GaussianBlur(img, (15, 15), 0)

    # 方法2：双边滤波磨皮（边缘保持清晰）
    bilateral_skin = cv2.bilateralFilter(img, 9, 75, 75)

    # 方法3：强力双边磨皮
    bilateral_strong = cv2.bilateralFilter(img, 15, 100, 100)

    # 方法4：迭代双边滤波（更强的磨皮效果，但更慢）
    bilateral_iter = img.copy()
    for _ in range(3):
        bilateral_iter = cv2.bilateralFilter(bilateral_iter, 5, 50, 50)

    # 显示
    cv2.imshow("Original", img)
    cv2.imshow("Gaussian Smooth", gaussian_skin)
    cv2.imshow("Bilateral Smooth", bilateral_skin)
    cv2.imshow("Bilateral Strong", bilateral_strong)
    cv2.imshow("Bilateral Iterative (3x)", bilateral_iter)

    print("人像磨皮效果对比")
    print("=" * 40)
    print("- 高斯滤波：皮肤平滑，但边缘（五官）也模糊了")
    print("- 双边滤波：皮肤平滑，边缘保持清晰！")
    print("- 强力双边：更强的磨皮效果")
    print("- 迭代双边：多次处理，效果更自然")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
