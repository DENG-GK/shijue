# -*- coding: utf-8 -*-
"""
σ 参数对模糊效果的影响
固定核大小，改变 σ 值
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), -1)
        cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)
        print("未找到测试图片，使用生成的示例图像")

    # 固定核大小 (15,15)，改变 σ
    sigmas = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

    cv2.imshow("Original", img)

    for sigma in sigmas:
        result = cv2.GaussianBlur(img, (15, 15), sigma)
        cv2.imshow(f"sigma={sigma}", result)

    print("σ 参数对模糊效果的影响（核大小固定为 15×15）：")
    print("- σ=0.5  → 几乎没有模糊（权重集中在中心）")
    print("- σ=1.0  → 轻微模糊")
    print("- σ=3.0  → 明显模糊")
    print("- σ=5.0  → 很强的模糊")
    print("- σ=10.0 → 接近均值模糊（权重几乎均匀分布）")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
