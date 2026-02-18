"""
交互式Canny边缘检测参数调节工具
实时调整阈值观察效果
"""

import cv2
import numpy as np

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建测试图像"""
    img = np.zeros((400, 500), dtype=np.uint8)
    img[:] = 80

    # 各种形状
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (280, 100), 50, 180, -1)
    cv2.ellipse(img, (400, 100), (60, 40), 0, 0, 360, 220, -1)

    # 线条
    cv2.line(img, (50, 200), (200, 350), 200, 5)
    cv2.line(img, (200, 200), (50, 350), 200, 5)

    # 多边形
    pts = np.array([[300, 200], [250, 300], [300, 380], [400, 350], [420, 250]], np.int32)
    cv2.fillPoly(img, [pts], 180)

    # 添加噪声
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

# 全局变量
img = create_test_image()
threshold1 = 50
threshold2 = 150
blur_size = 5

def nothing(x):
    pass

def update_display():
    """更新显示"""
    global img, threshold1, threshold2, blur_size

    # 获取参数
    threshold1 = cv2.getTrackbarPos('Low Threshold', 'Canny Edge Detection')
    threshold2 = cv2.getTrackbarPos('High Threshold', 'Canny Edge Detection')
    blur_size = cv2.getTrackbarPos('Blur Size', 'Canny Edge Detection')

    # 确保blur_size是奇数且大于0
    if blur_size % 2 == 0:
        blur_size += 1
    if blur_size < 1:
        blur_size = 1

    # 高斯模糊
    blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

    # Canny边缘检测
    edges = cv2.Canny(blurred, threshold1, threshold2)

    # 计算边缘像素数量
    edge_count = np.sum(edges > 0)

    # 创建显示图像
    # 左边：原图，右边：边缘
    display = np.hstack([img, edges])
    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    # 添加文字信息
    cv2.putText(display, 'Original', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, f'Canny (t1={threshold1}, t2={threshold2})', (510, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, f'Blur: {blur_size}x{blur_size}', (510, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
    cv2.putText(display, f'Edge pixels: {edge_count}', (510, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

    # 显示阈值比例提示
    if threshold2 > 0:
        ratio = threshold2 / max(1, threshold1)
        color = (0, 255, 0) if 2 <= ratio <= 3 else (0, 0, 255)
        cv2.putText(display, f'Ratio: {ratio:.1f} (recommend 2-3)', (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Canny Edge Detection', display)

# ===================== 创建窗口 =====================

cv2.namedWindow('Canny Edge Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Canny Edge Detection', 1000, 450)

# 创建trackbar
cv2.createTrackbar('Low Threshold', 'Canny Edge Detection', 50, 300, nothing)
cv2.createTrackbar('High Threshold', 'Canny Edge Detection', 150, 300, nothing)
cv2.createTrackbar('Blur Size', 'Canny Edge Detection', 5, 31, nothing)

print("=" * 60)
print("Canny边缘检测参数调节工具")
print("=" * 60)
print("\n参数说明：")
print("  Low Threshold:  低阈值（弱边缘）")
print("  High Threshold: 高阈值（强边缘）")
print("  Blur Size:      高斯模糊核大小")
print("\n调节建议：")
print("  • 高阈值 = 2~3 × 低阈值")
print("  • 比例在2-3之间时显示绿色")
print("  • Blur Size 根据噪声程度调整")
print("\n按 'q' 或 ESC 退出")
print("按 's' 保存当前结果")
print("=" * 60)

# ===================== 主循环 =====================

while True:
    update_display()

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('s'):
        # 保存当前结果
        blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        edges = cv2.Canny(blurred, threshold1, threshold2)
        cv2.imwrite('canny_result.png', edges)
        print(f"\n结果已保存：canny_result.png")
        print(f"参数：Low={threshold1}, High={threshold2}, Blur={blur_size}")

cv2.destroyAllWindows()

print("\n程序已退出")
print(f"最终参数: Low={threshold1}, High={threshold2}, Blur={blur_size}")
