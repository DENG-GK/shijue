"""
示例5：边缘检测的实际应用示例
包括：物体轮廓提取、艺术效果、文档处理
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建模拟场景图像 =====================

def create_scene_image():
    """创建一个模拟的场景图像"""
    img = np.zeros((400, 500, 3), dtype=np.uint8)

    # 天空渐变背景
    for y in range(200):
        img[y, :] = [180 - y//3, 200 - y//4, 230 - y//5]

    # 地面
    img[200:, :] = [50, 120, 50]

    # 太阳
    cv2.circle(img, (400, 60), 40, (0, 200, 255), -1)

    # 房子
    cv2.rectangle(img, (100, 180), (250, 350), (80, 80, 150), -1)  # 墙
    pts = np.array([[100, 180], [175, 100], [250, 180]], np.int32)
    cv2.fillPoly(img, [pts], (50, 50, 120))  # 屋顶
    cv2.rectangle(img, (140, 250), (210, 350), (50, 100, 150), -1)  # 门
    cv2.rectangle(img, (120, 200), (155, 235), (200, 200, 100), -1)  # 窗户
    cv2.rectangle(img, (195, 200), (230, 235), (200, 200, 100), -1)  # 窗户

    # 树
    cv2.rectangle(img, (330, 250), (360, 350), (40, 80, 100), -1)  # 树干
    cv2.circle(img, (345, 200), 60, (30, 100, 30), -1)  # 树冠

    # 云
    cv2.ellipse(img, (80, 50), (50, 20), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (100, 40), (30, 15), 0, 0, 360, (255, 255, 255), -1)

    return img

# 创建场景图像
scene_img = create_scene_image()
gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)

# ===================== 应用1：物体轮廓提取 =====================

# 使用Canny提取轮廓
edges = cv2.Canny(gray, 50, 150)

# 在原图上绘制轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
scene_with_contours = scene_img.copy()
cv2.drawContours(scene_with_contours, contours, -1, (0, 255, 0), 2)

# ===================== 应用2：卡通效果 =====================

def create_cartoon_effect(img):
    """创建卡通效果"""
    # 1. 边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 9)

    # 2. 颜色简化（双边滤波）
    color = cv2.bilateralFilter(img, 9, 300, 300)

    # 3. 合并边缘和颜色
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon

cartoon = create_cartoon_effect(scene_img)

# ===================== 应用3：素描效果 =====================

def create_sketch_effect(img):
    """创建素描效果"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 反转
    inv = 255 - gray

    # 高斯模糊
    blur = cv2.GaussianBlur(inv, (21, 21), 0)

    # 颜色减淡
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    return sketch

sketch = create_sketch_effect(scene_img)

# ===================== 应用4：边缘叠加效果 =====================

def edge_overlay(img):
    """边缘叠加到原图"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 边缘膨胀使其更明显
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 将边缘叠加到原图（黑色边缘）
    result = img.copy()
    result[edges > 0] = [0, 0, 0]

    return result

edge_art = edge_overlay(scene_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始场景图像', fontsize=12)
axes[0, 0].axis('off')

# 边缘检测结果
axes[0, 1].imshow(edges, cmap='gray')
axes[0, 1].set_title('Canny边缘检测', fontsize=12)
axes[0, 1].axis('off')

# 轮廓提取
axes[0, 2].imshow(cv2.cvtColor(scene_with_contours, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('物体轮廓提取', fontsize=12)
axes[0, 2].axis('off')

# 卡通效果
axes[1, 0].imshow(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('卡通效果', fontsize=12)
axes[1, 0].axis('off')

# 素描效果
axes[1, 1].imshow(sketch, cmap='gray')
axes[1, 1].set_title('素描效果', fontsize=12)
axes[1, 1].axis('off')

# 边缘叠加
axes[1, 2].imshow(cv2.cvtColor(edge_art, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('边缘叠加效果', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle('边缘检测的实际应用', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('edge_detection_applications.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n应用示例图已保存为 'edge_detection_applications.png'")
