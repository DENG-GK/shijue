# -*- coding: utf-8 -*-
"""
不同大小卷积核的模糊效果对比
"""

import cv2
import numpy as np


def main():
    # 读取图像
    img = cv2.imread("../images/test.jpg")
    if img is None:
        # 创建示例图像
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), 2)
        cv2.circle(img, (150, 150), 50, (0, 255, 0), -1)
        cv2.putText(img, "Test", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("未找到测试图片，使用生成的示例图像")

    # 使用不同大小的均值核进行模糊
    sizes = [3, 5, 9, 15, 25]

    cv2.imshow("Original", img)

    for size in sizes:
        # 创建均值核
        kernel = np.ones((size, size), dtype=np.float32) / (size * size)
        # 应用卷积
        result = cv2.filter2D(img, -1, kernel)
        # 显示
        cv2.imshow(f"Kernel {size}x{size}", result)

    print("不同核大小的模糊效果：")
    print("- 3x3: 轻微模糊")
    print("- 5x5: 明显模糊")
    print("- 9x9: 较强模糊")
    print("- 15x15: 很强模糊")
    print("- 25x25: 严重模糊")
    print("\n核越大，模糊越强，但细节丢失也越多！")
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
