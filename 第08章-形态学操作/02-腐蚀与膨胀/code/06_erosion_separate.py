"""
示例6：使用腐蚀分离粘连的物体
在细胞计数、颗粒分析等场景很常用
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建粘连物体 =====================

def create_touching_objects():
    """创建粘连在一起的物体"""
    img = np.zeros((200, 300), dtype=np.uint8)

    # 多个粘连的圆形（模拟细胞）
    cv2.circle(img, (60, 100), 35, 255, -1)
    cv2.circle(img, (110, 100), 35, 255, -1)
    cv2.circle(img, (160, 100), 35, 255, -1)

    # 另一组粘连物体
    cv2.circle(img, (240, 70), 25, 255, -1)
    cv2.circle(img, (260, 100), 25, 255, -1)
    cv2.circle(img, (240, 130), 25, 255, -1)

    return img

# ===================== 腐蚀分离 =====================

touching = create_touching_objects()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 不同程度的腐蚀
eroded_1 = cv2.erode(touching, kernel, iterations=2)
eroded_2 = cv2.erode(touching, kernel, iterations=4)
eroded_3 = cv2.erode(touching, kernel, iterations=6)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

images = [touching, eroded_1, eroded_2, eroded_3]
titles = ['原图（粘连）', '腐蚀 2 次', '腐蚀 4 次', '腐蚀 6 次']

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.suptitle('腐蚀分离粘连物体', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('erosion_separate.png', dpi=150)
plt.show()

# ===================== 计数对比 =====================

def count_objects(img):
    """使用连通组件分析计数"""
    num_labels, labels = cv2.connectedComponents(img)
    return num_labels - 1  # 减去背景

print("\n物体计数结果：")
print(f"• 原图（粘连）: {count_objects(touching)} 个物体")
print(f"• 腐蚀 2 次: {count_objects(eroded_1)} 个物体")
print(f"• 腐蚀 4 次: {count_objects(eroded_2)} 个物体")
print(f"• 腐蚀 6 次: {count_objects(eroded_3)} 个物体")

print("\n应用场景：")
print("• 细胞计数：分离粘连的细胞后准确计数")
print("• 颗粒分析：分离接触的颗粒进行测量")
print("• 字符分割：分离粘连的字符进行 OCR")
