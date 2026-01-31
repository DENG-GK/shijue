# -*- coding: utf-8 -*-
"""
添加噪声并观察效果
演示高斯噪声和椒盐噪声
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
    # 计算噪声像素数量
    num_noise = int(h * w * amount)

    # 添加"盐"（白点）
    for _ in range(num_noise):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        if len(image.shape) == 3:
            noisy[y, x] = [255, 255, 255]
        else:
            noisy[y, x] = 255

    # 添加"胡椒"（黑点）
    for _ in range(num_noise):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        if len(image.shape) == 3:
            noisy[y, x] = [0, 0, 0]
        else:
            noisy[y, x] = 0

    return noisy


def main():
    # 读取灰度图像
    img = cv2.imread("../images/test.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 创建示例图像
        img = np.ones((200, 200), dtype=np.uint8) * 128
        cv2.circle(img, (100, 100), 50, 200, -1)
        print("未找到测试图片，使用生成的示例图像")

    # 添加不同类型的噪声
    gaussian_noisy = add_gaussian_noise(img, mean=0, sigma=25)
    salt_pepper_noisy = add_salt_pepper_noise(img, amount=0.05)

    # 显示原图和加噪结果
    cv2.imshow("Original", img)
    cv2.imshow("Gaussian Noise (sigma=25)", gaussian_noisy)
    cv2.imshow("Salt & Pepper Noise (5%)", salt_pepper_noisy)

    print("噪声类型演示：")
    print("1. 高斯噪声 - 像蒙了一层薄雾，整体有沙沙的感觉")
    print("2. 椒盐噪声 - 随机的黑白点，像撒了胡椒和盐")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
