"""
示例2：创建用于形态学测试的二值图像
包含各种形状、噪点、空洞等情况
"""

import cv2
import numpy as np

def create_test_image():
    """创建一个包含各种测试元素的二值图像"""

    # 创建黑色背景
    img = np.zeros((300, 400), dtype=np.uint8)

    # 绘制主要形状
    # 1. 矩形
    cv2.rectangle(img, (30, 30), (120, 120), 255, -1)

    # 2. 圆形
    cv2.circle(img, (200, 75), 50, 255, -1)

    # 3. 椭圆
    cv2.ellipse(img, (330, 75), (40, 30), 0, 0, 360, 255, -1)

    # 4. 带有缺口的矩形（模拟断裂）
    cv2.rectangle(img, (30, 150), (120, 250), 255, -1)
    cv2.rectangle(img, (60, 190), (90, 210), 0, -1)  # 内部空洞

    # 5. 两个靠近的圆（模拟粘连）
    cv2.circle(img, (175, 200), 35, 255, -1)
    cv2.circle(img, (230, 200), 35, 255, -1)

    # 6. 添加一些噪点
    noise_positions = [
        (280, 160), (300, 180), (320, 170),
        (290, 220), (310, 240), (350, 200),
        (340, 230), (360, 180), (370, 210)
    ]
    for pos in noise_positions:
        cv2.circle(img, pos, 3, 255, -1)

    return img

# 创建测试图像
test_img = create_test_image()

# 显示图像
cv2.imshow('Test Image for Morphology', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图像
cv2.imwrite('morphology_test.png', test_img)
print("测试图像已保存为 'morphology_test.png'")
