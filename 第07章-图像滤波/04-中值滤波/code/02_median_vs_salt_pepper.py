# -*- coding: utf-8 -*-
"""
中值滤波 vs 椒盐噪声（效果惊艳！）
对比不同滤波器对椒盐噪声的处理效果
"""

import cv2
import numpy as np


def add_salt_pepper_noise(image, amount=0.05):
    """添加椒盐噪声"""
    noisy = image.copy()
    h, w = image.shape[:2]
    num_noise = int(h * w * amount)

    # 盐（白点）
    for _ in range(num_noise):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        if len(image.shape) == 3:
            noisy[y, x] = [255, 255, 255]
        else:
            noisy[y, x] = 255

    # 胡椒（黑点）
    for _ in range(num_noise):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        if len(image.shape) == 3:
            noisy[y, x] = [0, 0, 0]
        else:
            noisy[y, x] = 0

    return noisy


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), (200, 200, 200), -1)
        cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)
        print("未找到测试图片，使用生成的示例图像")

    # 添加 5% 的椒盐噪声
    noisy = add_salt_pepper_noise(img, amount=0.05)

    # 用不同滤波器处理
    blur_result = cv2.blur(noisy, (5, 5))              # 均值滤波
    gauss_result = cv2.GaussianBlur(noisy, (5, 5), 0)  # 高斯滤波
    median_result = cv2.medianBlur(noisy, 5)           # 中值滤波

    # 显示对比
    cv2.imshow("Original", img)
    cv2.imshow("Salt-Pepper Noisy (5%)", noisy)
    cv2.imshow("Blur 5x5", blur_result)
    cv2.imshow("Gaussian 5x5", gauss_result)
    cv2.imshow("Median 5 (Best!)", median_result)

    print("椒盐噪声滤波效果对比")
    print("=" * 40)
    print("观察结果：")
    print("- 均值滤波：噪声变成了灰点，图像变模糊")
    print("- 高斯滤波：和均值类似，噪声没完全消除")
    print("- 中值滤波：噪声几乎完全消失！边缘还很清晰！")
    print()
    print("结论：椒盐噪声的克星是中值滤波！")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
