"""
示例10：实时仿射变换工具
- OpenCV trackbar控制旋转/缩放/错切/平移
- 实时预览变换效果
- 快捷键：r重置 q退出
"""
import cv2
import numpy as np
import os

# 创建测试图像
image = np.zeros((300, 400, 3), dtype=np.uint8)
image[:, :] = [220, 220, 220]
cv2.rectangle(image, (100, 75), (300, 225), (0, 128, 255), -1)
cv2.putText(image, 'AFFINE', (130, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
cv2.circle(image, (130, 105), 15, (255, 0, 0), -1)

h, w = image.shape[:2]
cx, cy = w // 2, h // 2

params = {
    'rotation': 180, 'scale_x': 100, 'scale_y': 100,
    'shear_x': 50, 'shear_y': 50, 'trans_x': 50, 'trans_y': 50,
}


def update_transform(x=None):
    angle = params['rotation'] - 180
    sx = max(0.1, params['scale_x'] / 100.0)
    sy = max(0.1, params['scale_y'] / 100.0)
    shx = (params['shear_x'] - 50) / 100.0
    shy = (params['shear_y'] - 50) / 100.0
    tx = params['trans_x'] - 50
    ty = params['trans_y'] - 50

    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    Sh = np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]], dtype=np.float32)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)
    rad = np.radians(angle)
    R = np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    T2 = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float32)

    M = T2 @ R @ S @ Sh @ T1
    result = cv2.warpAffine(image, M[:2, :], (w, h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(128, 128, 128))

    info = f"Rot:{angle:.0f} Sx:{sx:.2f} Sy:{sy:.2f} ShX:{shx:.2f} ShY:{shy:.2f}"
    cv2.putText(result, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imshow('Affine Transform Tool', result)


def create_interactive_tool():
    cv2.namedWindow('Affine Transform Tool', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Rotation', 'Affine Transform Tool', 180, 360,
                       lambda v: (params.update({'rotation': v}), update_transform()))
    cv2.createTrackbar('Scale X', 'Affine Transform Tool', 100, 200,
                       lambda v: (params.update({'scale_x': max(10, v)}), update_transform()))
    cv2.createTrackbar('Scale Y', 'Affine Transform Tool', 100, 200,
                       lambda v: (params.update({'scale_y': max(10, v)}), update_transform()))
    cv2.createTrackbar('Shear X', 'Affine Transform Tool', 50, 100,
                       lambda v: (params.update({'shear_x': v}), update_transform()))
    cv2.createTrackbar('Shear Y', 'Affine Transform Tool', 50, 100,
                       lambda v: (params.update({'shear_y': v}), update_transform()))

    update_transform()

    print("交互式仿射变换工具")
    print("  Rotation: 旋转角度")
    print("  Scale X/Y: 缩放")
    print("  Shear X/Y: 错切")
    print("  r=重置 q=退出")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            cv2.setTrackbarPos('Rotation', 'Affine Transform Tool', 180)
            cv2.setTrackbarPos('Scale X', 'Affine Transform Tool', 100)
            cv2.setTrackbarPos('Scale Y', 'Affine Transform Tool', 100)
            cv2.setTrackbarPos('Shear X', 'Affine Transform Tool', 50)
            cv2.setTrackbarPos('Shear Y', 'Affine Transform Tool', 50)
    cv2.destroyAllWindows()


# 取消注释运行交互工具
# create_interactive_tool()

print("交互式仿射变换工具代码已就绪")
print("取消最后一行注释即可运行")
