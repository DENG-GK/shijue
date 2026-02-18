"""
示例6：交互式Sobel/Scharr对比工具
实时观察两种算子的差异
"""

import cv2
import numpy as np

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建测试图像"""
    img = np.zeros((400, 500), dtype=np.uint8)
    img[:] = 80

    # 各种形状和角度的边缘
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)

    # 斜线
    cv2.line(img, (200, 50), (300, 150), 200, 5)
    cv2.line(img, (200, 150), (300, 50), 200, 5)

    # 圆
    cv2.circle(img, (400, 100), 50, 200, -1)

    # 六边形
    center = (100, 280)
    pts = []
    for i in range(6):
        angle = i * 60 + 30
        x = int(center[0] + 60 * np.cos(np.radians(angle)))
        y = int(center[1] + 60 * np.sin(np.radians(angle)))
        pts.append([x, y])
    cv2.fillPoly(img, [np.array(pts, np.int32)], 200)

    # 菱形
    pts2 = np.array([[280, 220], [340, 280], [280, 340], [220, 280]], np.int32)
    cv2.fillPoly(img, [pts2], 200)

    # 曲线
    for i in range(100):
        x = 380 + i
        y = int(280 + 40 * np.sin(i * 0.1))
        if x < 500:
            cv2.circle(img, (x, y), 2, 200, -1)

    return img

# 全局变量
img = create_test_image()
use_blur = 1
threshold_val = 50

def nothing(x):
    pass

def update_display():
    """更新显示"""
    global img, use_blur, threshold_val

    # 获取参数
    use_blur = cv2.getTrackbarPos('Blur', 'Sobel vs Scharr')
    threshold_val = cv2.getTrackbarPos('Threshold', 'Sobel vs Scharr')

    # 预处理
    if use_blur:
        processed = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        processed = img.copy()

    # Sobel检测
    sobel_x = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(processed, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)

    # Scharr检测
    scharr_x = cv2.Scharr(processed, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(processed, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr_mag = np.clip(scharr_mag, 0, 255).astype(np.uint8)

    # 二值化
    _, sobel_bin = cv2.threshold(sobel_mag, threshold_val, 255, cv2.THRESH_BINARY)
    _, scharr_bin = cv2.threshold(scharr_mag, threshold_val, 255, cv2.THRESH_BINARY)

    # 差异图
    diff = cv2.absdiff(sobel_mag, scharr_mag)
    diff_colored = cv2.applyColorMap(diff * 3, cv2.COLORMAP_HOT)

    # 组合显示
    row1 = np.hstack([
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(scharr_mag, cv2.COLOR_GRAY2BGR)
    ])
    row2 = np.hstack([
        diff_colored,
        cv2.cvtColor(sobel_bin, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(scharr_bin, cv2.COLOR_GRAY2BGR)
    ])
    display = np.vstack([row1, row2])

    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display, 'Original', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, 'Sobel', (510, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, 'Scharr', (1010, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, 'Difference', (10, 430), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, 'Sobel Binary', (510, 430), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, 'Scharr Binary', (1010, 430), font, 0.7, (255, 255, 255), 2)

    cv2.imshow('Sobel vs Scharr', display)

# ===================== 创建窗口 =====================

cv2.namedWindow('Sobel vs Scharr', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sobel vs Scharr', 1500, 850)

cv2.createTrackbar('Blur', 'Sobel vs Scharr', 1, 1, nothing)
cv2.createTrackbar('Threshold', 'Sobel vs Scharr', 50, 255, nothing)

print("=" * 60)
print("Sobel vs Scharr 对比工具")
print("=" * 60)
print("\n窗口说明：")
print("  上排：原图 | Sobel幅值 | Scharr幅值")
print("  下排：差异热力图 | Sobel二值 | Scharr二值")
print("\n参数调节：")
print("  Blur: 是否使用高斯模糊预处理")
print("  Threshold: 二值化阈值")
print("\n按 'q' 或 ESC 退出")
print("=" * 60)

# ===================== 主循环 =====================

while True:
    update_display()

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()
print("\n程序已退出")
