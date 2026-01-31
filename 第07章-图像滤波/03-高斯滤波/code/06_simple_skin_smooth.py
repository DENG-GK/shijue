# -*- coding: utf-8 -*-
"""
高斯滤波的实际应用——简单磨皮
"""

import cv2
import numpy as np


def main():
    # 读取人像图片
    img = cv2.imread("../images/portrait.jpg")
    if img is None:
        # 创建一个模拟的"人像"图片
        img = np.ones((300, 300, 3), dtype=np.uint8) * 200
        # 添加一些模拟的"瑕疵"
        for _ in range(50):
            y, x = np.random.randint(50, 250), np.random.randint(50, 250)
            color = np.random.randint(100, 180)
            cv2.circle(img, (x, y), 2, (color, color, color), -1)
        print("未找到人像图片，使用模拟图像演示")

    # 方法：高斯模糊 + 原图混合 = 简单磨皮效果
    # 思路：模糊版本消除了皮肤瑕疵，和原图混合保留整体轮廓

    # 1. 强高斯模糊
    blurred = cv2.GaussianBlur(img, (0, 0), 10)

    # 2. 按比例混合：保留一部分细节
    #    alpha=0.7 原图 + beta=0.3 模糊 = 轻度磨皮
    alpha = 0.7   # 原图权重（越大越清晰）
    beta = 0.3    # 模糊权重（越大越光滑）
    skin_smooth = cv2.addWeighted(img, alpha, blurred, beta, 0)

    # 不同程度的磨皮
    light_smooth = cv2.addWeighted(img, 0.8, blurred, 0.2, 0)
    heavy_smooth = cv2.addWeighted(img, 0.5, blurred, 0.5, 0)

    # 显示
    cv2.imshow("Original", img)
    cv2.imshow("Gaussian Blurred", blurred)
    cv2.imshow("Light Smooth (80:20)", light_smooth)
    cv2.imshow("Medium Smooth (70:30)", skin_smooth)
    cv2.imshow("Heavy Smooth (50:50)", heavy_smooth)

    print("简单磨皮效果演示")
    print("=" * 40)
    print("原理：原图 + 模糊图 按比例混合")
    print()
    print("- 轻度磨皮：原图80% + 模糊20%")
    print("- 中度磨皮：原图70% + 模糊30%")
    print("- 重度磨皮：原图50% + 模糊50%")
    print()
    print("提示：这只是非常简单的磨皮")
    print("专业磨皮会用双边滤波（保留边缘），下一节会学到！")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
