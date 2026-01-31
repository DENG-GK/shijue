# -*- coding: utf-8 -*-
"""
对噪声图像进行均值滤波
对比均值滤波对高斯噪声和椒盐噪声的效果
"""

import cv2
import numpy as np


def add_gaussian_noise(image, mean=0, sigma=25):
    """添加高斯噪声"""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float64)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.05):
    """添加椒盐噪声"""
    noisy = image.copy()
    h, w = image.shape[:2]
    num_noise = int(h * w * amount)
    # 盐（白点）
    for _ in range(num_noise):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 255
    # 胡椒（黑点）
    for _ in range(num_noise):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 0
    return noisy


def main():
    # 读取灰度图像
    img = cv2.imread("../images/test.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.ones((200, 200), dtype=np.uint8) * 128
        cv2.circle(img, (100, 100), 50, 200, -1)
        print("未找到测试图片，使用生成的示例图像")

    # ========== 对高斯噪声进行均值滤波 ==========
    gaussian_noisy = add_gaussian_noise(img, sigma=30)

    gauss_blur3 = cv2.blur(gaussian_noisy, (3, 3))
    gauss_blur5 = cv2.blur(gaussian_noisy, (5, 5))
    gauss_blur9 = cv2.blur(gaussian_noisy, (9, 9))

    cv2.imshow("Original", img)
    cv2.imshow("Gaussian Noisy", gaussian_noisy)
    cv2.imshow("Gaussian + Blur 3x3", gauss_blur3)
    cv2.imshow("Gaussian + Blur 5x5", gauss_blur5)
    cv2.imshow("Gaussian + Blur 9x9", gauss_blur9)

    print("高斯噪声 + 均值滤波：效果不错，噪声被平均掉了")
    print("按任意键继续查看椒盐噪声效果...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ========== 对椒盐噪声进行均值滤波 ==========
    sp_noisy = add_salt_pepper_noise(img, amount=0.05)

    sp_blur3 = cv2.blur(sp_noisy, (3, 3))
    sp_blur5 = cv2.blur(sp_noisy, (5, 5))
    sp_blur9 = cv2.blur(sp_noisy, (9, 9))

    cv2.imshow("Original", img)
    cv2.imshow("Salt-Pepper Noisy", sp_noisy)
    cv2.imshow("SP + Blur 3x3", sp_blur3)
    cv2.imshow("SP + Blur 5x5", sp_blur5)
    cv2.imshow("SP + Blur 9x9", sp_blur9)

    print("\n椒盐噪声 + 均值滤波：效果一般，黑白极端值拖偏了平均值")
    print("（椒盐噪声应该用中值滤波！）")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
