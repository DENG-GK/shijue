# -*- coding: utf-8 -*-
"""
中值滤波的边缘保持能力
"""

import cv2
import numpy as np


def main():
    # 创建一个有清晰边缘的测试图像
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 255  # 中间一个白色正方形

    # 添加椒盐噪声
    noisy = img.copy()
    for _ in range(500):
        y, x = np.random.randint(0, 200), np.random.randint(0, 200)
        noisy[y, x] = 255 if np.random.random() > 0.5 else 0

    # 不同滤波处理
    blur = cv2.blur(noisy, (5, 5))
    gauss = cv2.GaussianBlur(noisy, (5, 5), 0)
    median = cv2.medianBlur(noisy, 5)

    # 计算边缘（用于对比边缘保持效果）
    edge_orig = cv2.Canny(img, 100, 200)
    edge_blur = cv2.Canny(blur, 100, 200)
    edge_gauss = cv2.Canny(gauss, 100, 200)
    edge_median = cv2.Canny(median, 100, 200)

    # 显示滤波结果
    cv2.imshow("Original", img)
    cv2.imshow("Noisy", noisy)
    cv2.imshow("Blur 5x5", blur)
    cv2.imshow("Gaussian 5x5", gauss)
    cv2.imshow("Median 5", median)

    print("按任意键查看边缘检测结果...")
    cv2.waitKey(0)

    # 显示边缘检测结果
    cv2.imshow("Edge - Original", edge_orig)
    cv2.imshow("Edge - Blur", edge_blur)
    cv2.imshow("Edge - Gaussian", edge_gauss)
    cv2.imshow("Edge - Median", edge_median)

    print("\n边缘保持能力对比：")
    print("- 均值滤波：边缘变得模糊、断断续续")
    print("- 高斯滤波：边缘略模糊")
    print("- 中值滤波：边缘保持最清晰！")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
