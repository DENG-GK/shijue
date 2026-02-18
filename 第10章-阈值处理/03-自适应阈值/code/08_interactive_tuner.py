"""
交互式自适应阈值参数调整工具
使用OpenCV trackbar实时调节blockSize和C参数
"""

import cv2
import numpy as np

def nothing(x):
    pass

# ===================== 创建测试图像 =====================

def create_test_image():
    """创建光照不均匀的测试图像"""
    img = np.ones((400, 600), dtype=np.uint8) * 220

    texts = ["OpenCV Tutorial", "Image Processing", "Thresholding", "Adaptive Method"]
    for i, text in enumerate(texts):
        y = 80 + i * 80
        cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 40, 2)

    rows, cols = img.shape
    for j in range(cols):
        factor = 1.0 - 0.5 * (j / cols)
        img[:, j] = (img[:, j] * factor).astype(np.uint8)

    return img

# ===================== 交互式参数调整 =====================

def interactive_adaptive_threshold(image):
    """交互式自适应阈值参数调整工具"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    cv2.namedWindow('Adaptive Threshold Tuner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Adaptive Threshold Tuner', 1200, 400)
    cv2.createTrackbar('blockSize', 'Adaptive Threshold Tuner', 11, 100, nothing)
    cv2.createTrackbar('C', 'Adaptive Threshold Tuner', 10, 50, nothing)
    cv2.createTrackbar('Method', 'Adaptive Threshold Tuner', 1, 1, nothing)

    print("=" * 50)
    print("自适应阈值参数调整工具")
    print("=" * 50)
    print("\n调参说明：")
    print("- blockSize: 邻域大小（会自动转为奇数）")
    print("- C: 从均值减去的常数")
    print("- Method: 0=MEAN, 1=GAUSSIAN")
    print("\n按 ESC 退出")
    print("按 's' 保存当前结果")

    while True:
        block_size = cv2.getTrackbarPos('blockSize', 'Adaptive Threshold Tuner')
        c_value = cv2.getTrackbarPos('C', 'Adaptive Threshold Tuner')
        method = cv2.getTrackbarPos('Method', 'Adaptive Threshold Tuner')

        # 确保blockSize是奇数且至少为3
        block_size = max(3, block_size)
        if block_size % 2 == 0:
            block_size += 1

        # 选择方法
        if method == 0:
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
            method_name = "MEAN"
        else:
            adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            method_name = "GAUSSIAN"

        # 应用自适应阈值
        result = cv2.adaptiveThreshold(gray, 255, adaptive_method,
                                       cv2.THRESH_BINARY, block_size, c_value)

        # 添加参数信息
        info = f"blockSize={block_size}, C={c_value}, Method={method_name}"
        display = result.copy()
        cv2.putText(display, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 128, 2)

        # 显示原图和结果
        combined = np.hstack([gray, display])
        cv2.imshow('Adaptive Threshold Tuner', combined)

        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            cv2.imwrite('adaptive_result.png', result)
            print(f"\n结果已保存: adaptive_result.png")
            print(f"参数: blockSize={block_size}, C={c_value}, Method={method_name}")

    cv2.destroyAllWindows()
    return block_size, c_value

# ===================== 主程序 =====================

test_img = create_test_image()
best_blocksize, best_c = interactive_adaptive_threshold(test_img)

print(f"\n最终参数: blockSize={best_blocksize}, C={best_c}")
