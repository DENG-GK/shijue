"""
示例4：展示所有基本形态学操作的效果对比
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_test_shape():
    """创建一个简单的测试形状"""
    img = np.zeros((200, 200), dtype=np.uint8)

    # 绘制一个有噪点和空洞的形状
    cv2.circle(img, (100, 100), 60, 255, -1)  # 主圆
    cv2.circle(img, (100, 100), 15, 0, -1)    # 内部空洞

    # 添加一些小噪点
    for pos in [(30, 30), (170, 40), (40, 160), (165, 170)]:
        cv2.circle(img, pos, 5, 255, -1)

    return img

# 创建测试图像
original = create_test_shape()

# 创建结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 执行各种形态学操作
eroded = cv2.erode(original, kernel, iterations=1)      # 腐蚀
dilated = cv2.dilate(original, kernel, iterations=1)    # 膨胀
opened = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)    # 开运算
closed = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)   # 闭运算
gradient = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, kernel)  # 梯度
tophat = cv2.morphologyEx(original, cv2.MORPH_TOPHAT, kernel)  # 顶帽
blackhat = cv2.morphologyEx(original, cv2.MORPH_BLACKHAT, kernel)  # 黑帽

# 创建对比图
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('形态学操作对比', fontsize=16, fontweight='bold')

operations = [
    (original, '原图'),
    (eroded, '腐蚀\n(边缘收缩)'),
    (dilated, '膨胀\n(边缘扩张)'),
    (opened, '开运算\n(去除噪点)'),
    (closed, '闭运算\n(填补空洞)'),
    (gradient, '形态梯度\n(边缘提取)'),
    (tophat, '顶帽\n(提取亮细节)'),
    (blackhat, '黑帽\n(提取暗细节)')
]

for ax, (img, title) in zip(axes.flatten(), operations):
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('morphology_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("对比图已保存为 'morphology_comparison.png'")
