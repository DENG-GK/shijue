# -*- coding: utf-8 -*-
"""图形绘制示例 - 直接生成绘制效果图"""
import cv2, numpy as np, os
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

img = np.ones((500, 700, 3), dtype=np.uint8) * 40  # 深灰背景

# 直线
cv2.line(img, (50, 50), (200, 50), (0, 255, 0), 2)
cv2.putText(img, "line", (210, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

# 箭头线
cv2.arrowedLine(img, (50, 100), (200, 100), (0, 255, 255), 2)
cv2.putText(img, "arrowedLine", (210, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

# 矩形
cv2.rectangle(img, (50, 140), (200, 220), (255, 0, 0), 2)
cv2.rectangle(img, (50, 240), (200, 320), (255, 0, 0), -1)  # 填充
cv2.putText(img, "rectangle", (210, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
cv2.putText(img, "filled", (210, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

# 圆
cv2.circle(img, (500, 80), 50, (0, 0, 255), 2)
cv2.circle(img, (500, 200), 50, (0, 0, 255), -1)
cv2.putText(img, "circle", (570, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

# 椭圆
cv2.ellipse(img, (500, 330), (80, 40), 30, 0, 360, (255, 255, 0), 2)
cv2.putText(img, "ellipse", (590, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

# 多边形
pts = np.array([[100, 360], [50, 450], [150, 480], [200, 420], [160, 370]], np.int32)
cv2.polylines(img, [pts], True, (255, 0, 255), 2)
cv2.putText(img, "polylines", (210, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

# 文字
cv2.putText(img, "OpenCV Drawing!", (350, 450), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

cv2.imwrite(os.path.join(IMAGES_DIR, "shapes.png"), img)
print(f"绘制结果已保存: {os.path.join(IMAGES_DIR, 'shapes.png')}")
