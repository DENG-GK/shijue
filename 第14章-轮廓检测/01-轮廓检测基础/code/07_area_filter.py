"""
示例7：轮廓面积筛选
- 按面积过滤轮廓
- 保留目标大小的轮廓
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建含不同大小形状的图像
np.random.seed(42)
img = np.zeros((400, 500), dtype=np.uint8)

# 大形状
cv2.rectangle(img, (20, 20), (200, 180), 255, -1)
cv2.circle(img, (350, 100), 80, 255, -1)

# 中形状
cv2.rectangle(img, (30, 220), (120, 300), 255, -1)
cv2.circle(img, (200, 300), 35, 255, -1)

# 小形状（噪声点）
for _ in range(30):
    x, y = np.random.randint(0, 500), np.random.randint(0, 400)
    r = np.random.randint(2, 8)
    cv2.circle(img, (x, y), r, 255, -1)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 按面积分类
areas = [cv2.contourArea(cnt) for cnt in contours]
thresholds = [100, 1000, 5000]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('轮廓面积筛选', fontsize=14, fontweight='bold')

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title(f'原始图像\n{len(contours)} 个轮廓')
axes[0, 0].axis('off')

# 面积直方图
axes[0, 1].hist(areas, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
for t in thresholds:
    axes[0, 1].axvline(t, color='red', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('面积')
axes[0, 1].set_ylabel('数量')
axes[0, 1].set_title('面积分布')

# 面积排序
sorted_areas = sorted(areas, reverse=True)
axes[0, 2].barh(range(min(15, len(sorted_areas))), sorted_areas[:15],
                color='steelblue', edgecolor='black')
axes[0, 2].set_xlabel('面积')
axes[0, 2].set_ylabel('轮廓序号')
axes[0, 2].set_title('面积排序 (前15)')

# 不同阈值筛选
for idx, threshold in enumerate(thresholds):
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold]
    canvas = np.zeros((400, 500, 3), dtype=np.uint8)
    cv2.drawContours(canvas, filtered, -1, (0, 255, 0), 2)
    axes[1, idx].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[1, idx].set_title(f'面积 > {threshold}\n{len(filtered)} 个轮廓')
    axes[1, idx].axis('off')

    print(f"面积 > {threshold}: {len(filtered)} 个轮廓")

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_area_filter.png'), dpi=150, bbox_inches='tight')
plt.show()
