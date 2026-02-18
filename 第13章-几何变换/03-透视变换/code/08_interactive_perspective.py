"""
示例8：交互式透视校正
- 鼠标拖拽角点
- 实时透视变换预览
- 快捷键：c校正 r重置 q退出
"""
import cv2
import numpy as np
import os

# 创建带透视畸变的测试图像
image = np.ones((400, 500, 3), dtype=np.uint8) * 200
pts = np.array([[80, 60], [420, 40], [450, 360], [50, 380]], np.int32)
cv2.fillPoly(image, [pts], (255, 255, 255))
for i in range(10):
    y_left = 80 + i * 28
    y_right = 60 + i * 30
    cv2.line(image, (100, y_left), (400, y_right), (100, 100, 100), 1)
cv2.putText(image, 'CORRECT ME', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

corners = pts.astype(np.float32)
state = {'selected': -1, 'corners': corners.copy()}


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        dists = [np.sqrt((c[0] - x) ** 2 + (c[1] - y) ** 2) for c in state['corners']]
        nearest = np.argmin(dists)
        if dists[nearest] < 20:
            state['selected'] = nearest
    elif event == cv2.EVENT_MOUSEMOVE:
        if state['selected'] >= 0:
            state['corners'][state['selected']] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        state['selected'] = -1


def create_perspective_tool():
    cv2.namedWindow('Perspective Correction')
    cv2.setMouseCallback('Perspective Correction', mouse_callback)

    print("交互式透视校正工具")
    print("  拖拽角点调整")
    print("  c=校正 r=重置 q=退出")

    while True:
        display = image.copy()
        cv2.polylines(display, [state['corners'].astype(int)], True, (0, 255, 0), 2)
        for i, pt in enumerate(state['corners'].astype(int)):
            cv2.circle(display, tuple(pt), 10, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (pt[0] - 5, pt[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Perspective Correction', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            state['corners'] = corners.copy()
        elif key == ord('c'):
            src = state['corners']
            w1 = np.linalg.norm(src[0] - src[1])
            w2 = np.linalg.norm(src[2] - src[3])
            h1 = np.linalg.norm(src[0] - src[3])
            h2 = np.linalg.norm(src[1] - src[2])
            tw, th = int(max(w1, w2)), int(max(h1, h2))
            dst = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
            M = cv2.getPerspectiveTransform(src, dst)
            corrected = cv2.warpPerspective(image, M, (tw, th))
            cv2.imshow('Corrected', corrected)
    cv2.destroyAllWindows()


# 取消注释运行交互工具
# create_perspective_tool()

print("交互式透视校正工具代码已就绪")
print("取消最后一行注释即可运行")
