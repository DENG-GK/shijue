"""
示例8：多尺度边缘提取
- 拉普拉斯金字塔每层取绝对值 → 边缘图
- 加权合并为综合边缘图
- 与Canny、Sobel对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def build_laplacian_pyramid(image, levels=4):
    G = [image.astype(np.float64)]
    current = image.astype(np.float64)
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        G.append(current)
    L = []
    for i in range(levels - 1):
        expanded = cv2.pyrUp(G[i + 1])
        if expanded.shape != G[i].shape:
            expanded = cv2.resize(expanded, (G[i].shape[1], G[i].shape[0]))
        L.append(G[i] - expanded)
    L.append(G[-1])
    return L


def multiscale_edges(image, levels=4):
    """多尺度边缘提取"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    lap_pyr = build_laplacian_pyramid(gray, levels)
    edge_maps = []
    for i, level in enumerate(lap_pyr[:-1]):
        edges = np.abs(level)
        if edges.max() > 0:
            edges = (edges / edges.max() * 255).astype(np.uint8)
        else:
            edges = edges.astype(np.uint8)
        edge_maps.append(edges)

    # 加权合并
    combined = np.zeros_like(gray, dtype=np.float64)
    for i, edges in enumerate(edge_maps):
        resized = cv2.resize(edges, (gray.shape[1], gray.shape[0]))
        weight = 1.0 / (i + 1)
        combined += resized * weight
    if combined.max() > 0:
        combined = (combined / combined.max() * 255).astype(np.uint8)
    return edge_maps, combined


# 创建测试图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.rectangle(image, (30, 30), (226, 226), (150, 150, 150), 2)
cv2.circle(image, (128, 128), 60, (200, 200, 200), 2)
cv2.line(image, (50, 200), (206, 200), (100, 100, 100), 1)
for i in range(20):
    cv2.circle(image, (30 + i * 11, 230), 3, (200, 200, 200), -1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edge_maps, combined = multiscale_edges(image, levels=4)

canny = cv2.Canny(gray, 50, 150)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
sobel = (sobel / sobel.max() * 255).astype(np.uint8) if sobel.max() > 0 else sobel.astype(np.uint8)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('多尺度边缘提取', fontsize=14, fontweight='bold')

axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title('原图')
axes[0, 0].axis('off')

for i, edges in enumerate(edge_maps):
    axes[0, i + 1].imshow(edges, cmap='gray')
    axes[0, i + 1].set_title(f'Level {i} 边缘')
    axes[0, i + 1].axis('off')

axes[1, 0].imshow(combined, cmap='gray')
axes[1, 0].set_title('多尺度合并')
axes[1, 0].axis('off')

axes[1, 1].imshow(canny, cmap='gray')
axes[1, 1].set_title('Canny')
axes[1, 1].axis('off')

axes[1, 2].imshow(sobel, cmap='gray')
axes[1, 2].set_title('Sobel')
axes[1, 2].axis('off')

# 对比展示
axes[1, 3].imshow(np.hstack([combined, canny, sobel]), cmap='gray')
axes[1, 3].set_title('多尺度 | Canny | Sobel')
axes[1, 3].axis('off')

axes[1, 4].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_multiscale_edges.png'), dpi=150, bbox_inches='tight')
plt.show()
