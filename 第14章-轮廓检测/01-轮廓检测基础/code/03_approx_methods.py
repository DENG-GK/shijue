"""
示例3：轮廓近似方法
- CHAIN_APPROX_NONE vs CHAIN_APPROX_SIMPLE
- 点数差异对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (30, 30), (180, 150), 255, -1)
cv2.circle(img, (300, 100), 60, 255, -1)
pts = np.array([[100, 200], [200, 180], [250, 280], [50, 280]], np.int32)
cv2.fillPoly(img, [pts], 255)

methods = {
    'CHAIN_APPROX_NONE': cv2.CHAIN_APPROX_NONE,
    'CHAIN_APPROX_SIMPLE': cv2.CHAIN_APPROX_SIMPLE,
    'CHAIN_APPROX_TC89_L1': cv2.CHAIN_APPROX_TC89_L1,
    'CHAIN_APPROX_TC89_KCOS': cv2.CHAIN_APPROX_TC89_KCOS,
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('轮廓近似方法对比', fontsize=14, fontweight='bold')

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始二值图像')
axes[0, 0].axis('off')

# 放大区域
axes[0, 1].imshow(img[20:160, 20:190], cmap='gray')
axes[0, 1].set_title('矩形放大')
axes[0, 1].axis('off')

# 统计信息
summary = "近似方法\t\t   总点数\n" + "-" * 40 + "\n"

for idx, (name, method) in enumerate(methods.items()):
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, method)
    total_points = sum(len(cnt) for cnt in contours)
    summary += f"{name}:  {total_points}\n"

    canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 1)
        for pt in cnt:
            cv2.circle(canvas, tuple(pt[0]), 2, (0, 0, 255), -1)

    row = (idx + 2) // 3
    col = (idx + 2) % 3
    axes[row, col].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f'{name}\n总点数: {total_points}')
    axes[row, col].axis('off')

print(summary)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_approx_methods.png'), dpi=150, bbox_inches='tight')
plt.show()
