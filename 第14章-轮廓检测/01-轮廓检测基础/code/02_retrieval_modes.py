"""
示例2：轮廓检索模式
- RETR_EXTERNAL / RETR_LIST / RETR_CCOMP / RETR_TREE
- 不同模式下的轮廓层次结构
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建含嵌套形状的图像
img = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img, (20, 20), (280, 280), 255, -1)      # 外矩形
cv2.rectangle(img, (50, 50), (250, 250), 0, -1)         # 内矩形（挖空）
cv2.circle(img, (150, 150), 60, 255, -1)                 # 圆（在中间）
cv2.circle(img, (150, 150), 30, 0, -1)                   # 小圆（挖空）

modes = {
    'RETR_EXTERNAL': cv2.RETR_EXTERNAL,
    'RETR_LIST': cv2.RETR_LIST,
    'RETR_CCOMP': cv2.RETR_CCOMP,
    'RETR_TREE': cv2.RETR_TREE,
}

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle('轮廓检索模式对比', fontsize=14, fontweight='bold')

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始二值图像')
axes[0].axis('off')

colors_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
               (255, 0, 255), (0, 255, 255)]

for idx, (name, mode) in enumerate(modes.items()):
    contours, hierarchy = cv2.findContours(img.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
    canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    for i, cnt in enumerate(contours):
        cv2.drawContours(canvas, [cnt], 0, colors_list[i % len(colors_list)], 2)

    axes[idx + 1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[idx + 1].set_title(f'{name}\n{len(contours)} 个轮廓')
    axes[idx + 1].axis('off')

    print(f"\n{name}: {len(contours)} 个轮廓")
    if hierarchy is not None:
        for i in range(len(contours)):
            h = hierarchy[0][i]
            print(f"  轮廓{i}: [next={h[0]}, prev={h[1]}, child={h[2]}, parent={h[3]}]")

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_retrieval_modes.png'), dpi=150, bbox_inches='tight')
plt.show()
