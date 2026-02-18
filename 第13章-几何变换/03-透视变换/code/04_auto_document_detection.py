"""
示例4：自动文档角点检测
- Canny边缘检测
- findContours + approxPolyDP
- 自动检测最大四边形
- 透视校正
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建场景（桌面+文档）
canvas = np.ones((600, 800, 3), dtype=np.uint8) * 150
for i in range(0, 800, 10):
    intensity = 140 + np.random.randint(-10, 10)
    cv2.line(canvas, (i, 0), (i, 599), (intensity, intensity - 10, intensity - 20), 2)

# 创建文档
doc = np.ones((350, 250, 3), dtype=np.uint8) * 252
cv2.rectangle(doc, (15, 15), (235, 335), (0, 0, 0), 1)
for i in range(12):
    y = 40 + i * 25
    cv2.line(doc, (30, y), (220, y), (150, 150, 150), 1)
cv2.putText(doc, 'INVOICE', (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

# 透视放置文档
dh, dw = doc.shape[:2]
src_pts = np.float32([[0, 0], [dw, 0], [dw, dh], [0, dh]])
dst_pts = np.float32([[150, 100], [550, 80], [600, 480], [100, 520]])

M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped_doc = cv2.warpPerspective(doc, M, (800, 600))
mask = cv2.warpPerspective(np.ones((dh, dw), dtype=np.uint8) * 255, M, (800, 600))
canvas[mask > 0] = warped_doc[mask > 0]


def detect_document_corners(img):
    """自动检测文档角点"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_corners = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                max_area = area
                best_corners = approx.reshape(4, 2)
    return best_corners


def order_corners(pts):
    """排序角点：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


detected = detect_document_corners(canvas)
corrected = None

if detected is not None:
    ordered = order_corners(detected)
    w1 = np.linalg.norm(ordered[0] - ordered[1])
    w2 = np.linalg.norm(ordered[2] - ordered[3])
    h1 = np.linalg.norm(ordered[0] - ordered[3])
    h2 = np.linalg.norm(ordered[1] - ordered[2])
    tw, th = int(max(w1, w2)), int(max(h1, h2))
    target_pts = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    M_correct = cv2.getPerspectiveTransform(ordered, target_pts)
    corrected = cv2.warpPerspective(canvas, M_correct, (tw, th))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('自动文档角点检测与校正', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('输入图像')
axes[0, 0].axis('off')

gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
axes[0, 1].imshow(edges, cmap='gray')
axes[0, 1].set_title('Canny边缘检测')
axes[0, 1].axis('off')

canvas_corners = canvas.copy()
if detected is not None:
    for i, pt in enumerate(ordered.astype(int)):
        cv2.circle(canvas_corners, tuple(pt), 10, (0, 255, 0), -1)
        cv2.putText(canvas_corners, str(i + 1), (pt[0] + 15, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.polylines(canvas_corners, [ordered.astype(int)], True, (0, 255, 0), 2)
axes[0, 2].imshow(cv2.cvtColor(canvas_corners, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('检测到的角点')
axes[0, 2].axis('off')

if corrected is not None:
    axes[1, 0].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'校正结果\n{corrected.shape[1]}x{corrected.shape[0]}')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(doc, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('原始文档')
axes[1, 1].axis('off')

# 处理流程
pipeline = """检测流程:

1. 灰度化 + 高斯模糊
2. Canny边缘检测
3. 查找轮廓 findContours
4. 多边形近似 approxPolyDP
5. 选择最大四边形
6. 角点排序
7. 计算透视变换矩阵
8. 透视校正 warpPerspective"""
axes[1, 2].text(0.05, 0.5, pipeline, fontsize=9, family='monospace',
                verticalalignment='center', transform=axes[1, 2].transAxes)
axes[1, 2].axis('off')
axes[1, 2].set_title('处理流程')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_auto_document_detection.png'), dpi=150, bbox_inches='tight')
plt.show()

print("自动文档检测与校正完成！")
