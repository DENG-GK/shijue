"""
示例5：交互式阈值选择（使用滑动条）
"""
import cv2
import numpy as np

def nothing(x):
    pass

def interactive_threshold():
    """交互式阈值调整工具"""
    img = np.zeros((300, 400), dtype=np.uint8)
    cv2.circle(img, (200, 150), 100, 180, -1)
    cv2.rectangle(img, (50, 50), (150, 250), 120, -1)
    noise = np.random.normal(0, 20, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    cv2.namedWindow('Threshold Adjustment')
    cv2.createTrackbar('Threshold', 'Threshold Adjustment', 127, 255, nothing)
    cv2.createTrackbar('Type', 'Threshold Adjustment', 0, 4, nothing)

    type_names = ['BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    types = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC,
             cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]

    while True:
        thresh_val = cv2.getTrackbarPos('Threshold', 'Threshold Adjustment')
        type_idx = cv2.getTrackbarPos('Type', 'Threshold Adjustment')

        _, result = cv2.threshold(img, thresh_val, 255, types[type_idx])

        display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        cv2.putText(display, f'T={thresh_val}, Type={type_names[type_idx]}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        original_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([original_bgr, display])
        cv2.imshow('Threshold Adjustment', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()
    return thresh_val

print("提示：运行 interactive_threshold() 函数可以交互式调整阈值")
print("使用滑动条调整阈值和阈值类型，按ESC退出")

# 取消注释下面的行来运行交互式工具
# threshold = interactive_threshold()
