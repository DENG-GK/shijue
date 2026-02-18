"""
示例6：轮廓掩码
- 用轮廓创建掩码
- 提取轮廓内的图像区域
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建彩色图像
np.random.seed(42)
img = np.ones((300, 400, 3), dtype=np.uint8) * 200
for _ in range(500):
    x, y = np.random.randint(0, 400), np.random.randint(0, 300)
    color = tuple(np.random.randint(50, 200, 3).tolist())
    cv2.circle(img, (x, y), np.random.randint(2, 8), color, -1)

# 创建形状二值图
binary = np.zeros((300, 400), dtype=np.uint8)
cv2.circle(binary, (120, 150), 80, 255, -1)
cv2.rectangle(binary, (250, 50), (380, 250), 255, -1)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('轮廓掩码', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(binary, cmap='gray')
axes[0, 1].set_title('轮廓形状')
axes[0, 1].axis('off')

# 为每个轮廓创建掩码并提取
for i, cnt in enumerate(contours):
    mask = np.zeros((300, 400), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)

    # 应用掩码
    result = cv2.bitwise_and(img, img, mask=mask)

    axes[0, 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) if i == 0 else None

# 合并所有掩码
full_mask = np.zeros((300, 400), dtype=np.uint8)
cv2.drawContours(full_mask, contours, -1, 255, -1)
masked_result = cv2.bitwise_and(img, img, mask=full_mask)
axes[0, 2].imshow(cv2.cvtColor(masked_result, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('掩码提取结果')
axes[0, 2].axis('off')

# 反向掩码
inv_mask = cv2.bitwise_not(full_mask)
inv_result = cv2.bitwise_and(img, img, mask=inv_mask)
axes[1, 0].imshow(cv2.cvtColor(inv_result, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('反向掩码')
axes[1, 0].axis('off')

# 掩码边界
edge_mask = np.zeros((300, 400), dtype=np.uint8)
cv2.drawContours(edge_mask, contours, -1, 255, 3)
edge_result = cv2.bitwise_and(img, img, mask=edge_mask)
axes[1, 1].imshow(cv2.cvtColor(edge_result, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('边界掩码')
axes[1, 1].axis('off')

# 叠加效果
overlay = img.copy()
colored = np.zeros_like(img)
cv2.drawContours(colored, contours, -1, (0, 200, 0), -1)
overlay = cv2.addWeighted(overlay, 0.7, colored, 0.3, 0)
cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('半透明叠加')
axes[1, 2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_contour_mask.png'), dpi=150, bbox_inches='tight')
plt.show()
