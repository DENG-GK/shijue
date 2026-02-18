"""
示例7：交互式CLAHE参数调整
- 使用OpenCV trackbar实时调整clipLimit和tileSize
- 实时预览CLAHE效果
- 适合实际调参使用（需要GUI环境）
"""
import cv2
import numpy as np
import os


def nothing(x):
    pass


def interactive_clahe(image=None):
    """交互式CLAHE参数调整工具"""
    if image is None:
        # 创建默认测试图像
        image = np.zeros((300, 400), dtype=np.uint8)
        image[:, :200] = np.random.normal(60, 15, (300, 200)).clip(0, 255)
        image[:, 200:] = np.random.normal(180, 15, (300, 200)).clip(0, 255)
        cv2.rectangle(image, (50, 50), (150, 200), 100, -1)
        cv2.circle(image, (300, 150), 60, 140, -1)
        image = image.astype(np.uint8)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    cv2.namedWindow('CLAHE Tuner')
    cv2.createTrackbar('clipLimit', 'CLAHE Tuner', 20, 100, nothing)  # *0.1
    cv2.createTrackbar('tileSize', 'CLAHE Tuner', 8, 32, nothing)

    print("CLAHE参数调整工具")
    print("- clipLimit: 对比度限制 (显示值/10)")
    print("- tileSize: 分块大小")
    print("按ESC退出")

    while True:
        clip = cv2.getTrackbarPos('clipLimit', 'CLAHE Tuner') / 10.0
        tile = cv2.getTrackbarPos('tileSize', 'CLAHE Tuner')

        if clip < 0.1:
            clip = 0.1
        if tile < 2:
            tile = 2

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        result = clahe.apply(gray)

        # 添加参数信息
        display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        info = f"clipLimit={clip:.1f}, tileSize=({tile},{tile})"
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('CLAHE Tuner', display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    return clip, tile


if __name__ == '__main__':
    print("=" * 50)
    print("交互式CLAHE参数调整工具")
    print("=" * 50)
    print("提示：运行此脚本将打开交互式窗口")
    print("- 拖动clipLimit滑条调整对比度限制")
    print("- 拖动tileSize滑条调整分块大小")
    print("- 按ESC退出")
    print()

    best_clip, best_tile = interactive_clahe()
    print(f"\n最终选择的参数：")
    print(f"  clipLimit = {best_clip:.1f}")
    print(f"  tileSize = ({best_tile}, {best_tile})")
