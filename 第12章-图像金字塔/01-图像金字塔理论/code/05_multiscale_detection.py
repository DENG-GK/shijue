"""
示例5：多尺度目标检测概念
- 不同大小的目标模拟
- 金字塔多尺度搜索
- 固定检测窗口在不同层级检测不同大小目标
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建包含不同大小目标的场景
image = np.ones((400, 600, 3), dtype=np.uint8) * 240

# 大目标（近处）
cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)
cv2.putText(image, 'L', (85, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

# 中等目标
cv2.rectangle(image, (250, 100), (310, 160), (0, 255, 0), -1)
cv2.putText(image, 'M', (265, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# 小目标（远处）
cv2.rectangle(image, (450, 130), (480, 160), (0, 0, 255), -1)

# 更小的目标
cv2.rectangle(image, (520, 140), (535, 155), (255, 0, 255), -1)

# 构建金字塔
pyramid = [image]
current = image
for i in range(4):
    current = cv2.pyrDown(current)
    pyramid.append(current)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('多尺度目标检测概念', fontsize=14, fontweight='bold')

# 各层级
for i, level in enumerate(pyramid):
    axes[0, i].imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'Level {i}\n{level.shape[1]}×{level.shape[0]}')
    axes[0, i].axis('off')

# 检测结果标注
result = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
detections = [
    ((50, 50, 150, 150), 'Level 0', (255, 0, 0)),
    ((250, 100, 310, 160), 'Level 1', (0, 200, 0)),
    ((450, 130, 480, 160), 'Level 2', (0, 0, 255)),
    ((520, 140, 535, 155), 'Level 3', (200, 0, 200))
]
for (x1, y1, x2, y2), label, color in detections:
    cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
    cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

axes[1, 0].imshow(result)
axes[1, 0].set_title('综合检测结果')
axes[1, 0].axis('off')

# 说明文字
explanation = (
    "多尺度检测原理：\n"
    "1. 构建图像金字塔\n"
    "2. 在每层用固定大小检测器\n"
    "3. 不同层检测不同大小目标\n"
    "4. 合并各层检测结果\n\n"
    "优点：\n"
    "- 无需调整检测器大小\n"
    "- 自然处理尺度变化"
)
axes[1, 1].text(0.1, 0.5, explanation, fontsize=9, va='center',
                transform=axes[1, 1].transAxes, family='SimHei')
axes[1, 1].axis('off')
axes[1, 1].set_title('原理说明')

for j in range(2, 5):
    axes[1, j].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_multiscale_detection.png'), dpi=150, bbox_inches='tight')
plt.show()
