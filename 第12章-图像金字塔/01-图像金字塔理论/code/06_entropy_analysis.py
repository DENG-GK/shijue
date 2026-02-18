"""
示例6：金字塔信息熵分析
- 各层信息熵变化
- 边缘密度随层级递减
- 对比度（标准差）随层级变化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_entropy(image):
    """计算图像信息熵"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def calculate_edge_density(image):
    """计算边缘密度"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges > 0) / edges.size * 100


# 创建多尺度细节图像
image = np.zeros((512, 512, 3), dtype=np.uint8)
# 粗特征
for i in range(4):
    for j in range(4):
        color = np.random.randint(50, 200, 3).tolist()
        cv2.rectangle(image, (i * 128, j * 128), ((i + 1) * 128, (j + 1) * 128), color, -1)
# 中等特征
for i in range(16):
    for j in range(16):
        if (i + j) % 2 == 0:
            cv2.circle(image, (i * 32 + 16, j * 32 + 16), 10,
                       (np.random.randint(200, 255), np.random.randint(200, 255),
                        np.random.randint(200, 255)), -1)
# 细节纹理
noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
image = cv2.add(image, noise)

# 构建金字塔并分析
pyramid = [image]
current = image
for i in range(5):
    current = cv2.pyrDown(current)
    pyramid.append(current)

metrics = {
    'level': [], 'size': [], 'entropy': [],
    'edge_density': [], 'mean': [], 'std': []
}

for i, level in enumerate(pyramid):
    metrics['level'].append(i)
    metrics['size'].append(f"{level.shape[1]}×{level.shape[0]}")
    metrics['entropy'].append(calculate_entropy(level))
    metrics['edge_density'].append(calculate_edge_density(level))
    metrics['mean'].append(float(np.mean(level)))
    metrics['std'].append(float(np.std(level)))

fig = plt.figure(figsize=(18, 12))
fig.suptitle('金字塔信息熵与特征分析', fontsize=14, fontweight='bold')

# 金字塔各层
for i, level in enumerate(pyramid):
    ax = fig.add_subplot(3, 6, i + 1)
    ax.imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Level {i}\n{metrics["size"][i]}', fontsize=9)
    ax.axis('off')

# 信息熵
ax1 = fig.add_subplot(3, 3, 7)
ax1.plot(metrics['level'], metrics['entropy'], 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('金字塔层级')
ax1.set_ylabel('信息熵 (bits)')
ax1.set_title('信息熵随层级变化')
ax1.grid(True, alpha=0.3)

# 边缘密度
ax2 = fig.add_subplot(3, 3, 8)
ax2.plot(metrics['level'], metrics['edge_density'], 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('金字塔层级')
ax2.set_ylabel('边缘密度 (%)')
ax2.set_title('边缘密度随层级变化')
ax2.grid(True, alpha=0.3)

# 标准差
ax3 = fig.add_subplot(3, 3, 9)
ax3.plot(metrics['level'], metrics['std'], 'go-', linewidth=2, markersize=8)
ax3.set_xlabel('金字塔层级')
ax3.set_ylabel('标准差')
ax3.set_title('对比度随层级变化')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_entropy_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n金字塔分析摘要:")
print(f"{'层级':<6} {'尺寸':<12} {'信息熵':<10} {'边缘%':<10} {'均值':<10} {'标准差':<10}")
print("-" * 60)
for i in range(len(pyramid)):
    print(f"{metrics['level'][i]:<6} {metrics['size'][i]:<12} "
          f"{metrics['entropy'][i]:.2f}      {metrics['edge_density'][i]:.2f}%     "
          f"{metrics['mean'][i]:.1f}      {metrics['std'][i]:.1f}")
