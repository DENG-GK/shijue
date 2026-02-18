"""
示例6：交互式边缘检测参数调节工具
使用OpenCV的trackbar实时调整参数
"""

import cv2
import numpy as np

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建测试图像"""
    img = np.zeros((400, 500), dtype=np.uint8)
    img[:] = 50

    # 各种形状
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (250, 100), 50, 180, -1)
    cv2.ellipse(img, (400, 100), (60, 40), 30, 0, 360, 220, -1)

    # 带渐变的区域
    for i in range(150):
        img[200:350, 50+i*2:52+i*2] = 50 + i

    # 一些线条
    cv2.line(img, (300, 200), (450, 350), 200, 3)
    cv2.line(img, (300, 350), (450, 200), 200, 3)

    return img

# 全局变量
img = create_test_image()
blur_size = 3
threshold1 = 50
threshold2 = 150

def nothing(x):
    """空回调函数"""
    pass

def update_edge_detection():
    """更新边缘检测结果"""
    global img, blur_size, threshold1, threshold2

    # 获取trackbar的值
    blur_size = cv2.getTrackbarPos('Blur Size', 'Edge Detection')
    threshold1 = cv2.getTrackbarPos('Threshold1', 'Edge Detection')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'Edge Detection')

    # 确保blur_size是奇数且大于0
    if blur_size % 2 == 0:
        blur_size += 1
    if blur_size < 1:
        blur_size = 1

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

    # 应用Canny边缘检测
    edges = cv2.Canny(blurred, threshold1, threshold2)

    # 创建对比显示
    display = np.hstack([img, blurred, edges])

    # 添加文字说明
    cv2.putText(display, 'Original', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    cv2.putText(display, f'Blurred (k={blur_size})', (510, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    cv2.putText(display, f'Canny ({threshold1}-{threshold2})', (1010, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

    cv2.imshow('Edge Detection', display)

# ===================== 创建窗口和trackbar =====================

cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Edge Detection', 1500, 450)

# 创建trackbar
cv2.createTrackbar('Blur Size', 'Edge Detection', 3, 31, nothing)
cv2.createTrackbar('Threshold1', 'Edge Detection', 50, 300, nothing)
cv2.createTrackbar('Threshold2', 'Edge Detection', 150, 300, nothing)

print("=" * 60)
print("边缘检测参数调节工具")
print("=" * 60)
print("\n使用说明：")
print("  • Blur Size: 高斯模糊核大小（越大越平滑）")
print("  • Threshold1: Canny低阈值（低于此值不是边缘）")
print("  • Threshold2: Canny高阈值（高于此值一定是边缘）")
print("\n参数调节建议：")
print("  • Threshold2 通常设为 Threshold1 的 2-3 倍")
print("  • Blur Size 根据噪声程度调整")
print("\n按 'q' 或 ESC 键退出")
print("=" * 60)

# ===================== 主循环 =====================

while True:
    update_edge_detection()

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q') or key == 27:  # 'q' 或 ESC 退出
        break

cv2.destroyAllWindows()

print("\n程序已退出")
print(f"最终参数: Blur={blur_size}, Threshold1={threshold1}, Threshold2={threshold2}")
