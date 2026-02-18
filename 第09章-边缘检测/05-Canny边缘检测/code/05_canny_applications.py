"""
Canny边缘检测的实际应用示例
包括：轮廓检测、文档边缘、物体检测等
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建模拟场景 =====================

def create_document_image():
    """创建模拟的文档图像"""
    img = np.ones((400, 300, 3), dtype=np.uint8) * 180  # 灰色背景

    # 白色文档
    pts = np.array([[50, 50], [250, 30], [270, 350], [30, 370]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    # 文字行（用矩形模拟）
    for y in range(80, 320, 30):
        width = np.random.randint(100, 180)
        cv2.rectangle(img, (70, y), (70+width, y+15), (50, 50, 50), -1)

    return img

def create_object_image():
    """创建模拟的物体图像"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # 渐变背景
    for y in range(300):
        img[y, :] = [80 + y//4, 100 + y//5, 120 + y//6]

    # 物体1：杯子
    cv2.ellipse(img, (100, 200), (40, 20), 0, 0, 360, (200, 180, 160), -1)
    cv2.rectangle(img, (60, 100), (140, 200), (200, 180, 160), -1)
    cv2.ellipse(img, (100, 100), (40, 15), 0, 0, 360, (220, 200, 180), -1)

    # 物体2：盒子
    pts = np.array([[200, 250], [200, 150], [280, 120], [350, 150],
                    [350, 250], [280, 280]], np.int32)
    cv2.fillPoly(img, [pts], (100, 150, 200))
    cv2.line(img, (280, 120), (280, 280), (80, 130, 180), 2)

    # 物体3：球
    cv2.circle(img, (320, 80), 35, (200, 100, 100), -1)

    return img

# 创建图像
doc_img = create_document_image()
obj_img = create_object_image()

# ===================== 应用1：文档边缘检测 =====================

doc_gray = cv2.cvtColor(doc_img, cv2.COLOR_BGR2GRAY)
doc_blur = cv2.GaussianBlur(doc_gray, (5, 5), 1.0)
doc_edges = cv2.Canny(doc_blur, 50, 150)

# 找轮廓
contours, _ = cv2.findContours(doc_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找最大轮廓（文档）
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    # 近似多边形
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 绘制结果
    doc_result = doc_img.copy()
    cv2.drawContours(doc_result, [approx], -1, (0, 255, 0), 3)

    # 标记角点
    for point in approx:
        cv2.circle(doc_result, tuple(point[0]), 8, (0, 0, 255), -1)

# ===================== 应用2：物体轮廓检测 =====================

obj_gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
obj_blur = cv2.GaussianBlur(obj_gray, (5, 5), 1.0)
obj_edges = cv2.Canny(obj_blur, 30, 100)

# 找轮廓
obj_contours, _ = cv2.findContours(obj_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制所有轮廓
obj_result = obj_img.copy()
cv2.drawContours(obj_result, obj_contours, -1, (0, 255, 0), 2)

# 为每个轮廓画边界框
for cnt in obj_contours:
    if cv2.contourArea(cnt) > 500:  # 过滤小轮廓
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(obj_result, (x, y), (x+w, y+h), (255, 0, 0), 2)

# ===================== 应用3：边缘叠加效果 =====================

def edge_overlay_effect(img, low_thresh=50, high_thresh=150):
    """边缘叠加艺术效果"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blur, low_thresh, high_thresh)

    # 膨胀边缘使其更明显
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 叠加到原图（黑色边缘）
    result = img.copy()
    result[edges > 0] = [0, 0, 0]

    return result, edges

obj_overlay, _ = edge_overlay_effect(obj_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 文档处理
axes[0, 0].imshow(cv2.cvtColor(doc_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始文档图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(doc_edges, cmap='gray')
axes[0, 1].set_title('Canny边缘', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(doc_result, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('文档轮廓检测\n（绿色边框，红色角点）', fontsize=11)
axes[0, 2].axis('off')

# 说明
axes[0, 3].axis('off')
info1 = """
文档边缘检测应用：

1. Canny检测边缘
2. 查找轮廓
3. 多边形近似
4. 提取文档四角

应用场景：
• 扫描仪应用
• 文档校正
• OCR预处理
"""
axes[0, 3].text(0.1, 0.5, info1, fontsize=9,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# 物体处理
axes[1, 0].imshow(cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('原始物体图像', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(obj_edges, cmap='gray')
axes[1, 1].set_title('Canny边缘', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(obj_result, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('物体轮廓检测\n（绿色轮廓，蓝色边界框）', fontsize=11)
axes[1, 2].axis('off')

axes[1, 3].imshow(cv2.cvtColor(obj_overlay, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title('边缘叠加效果\n（艺术效果）', fontsize=11)
axes[1, 3].axis('off')

plt.suptitle('Canny边缘检测实际应用', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('canny_applications.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n应用示例图已保存为 'canny_applications.png'")
