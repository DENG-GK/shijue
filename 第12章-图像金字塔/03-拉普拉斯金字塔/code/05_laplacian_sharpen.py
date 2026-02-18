"""
示例5：拉普拉斯金字塔锐化
- 增强细节层(×sharpen_factor)
- 不同factor对比效果
- 边缘响应对比
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


def reconstruct_from_laplacian(pyramid):
    result = pyramid[-1].copy()
    for i in range(len(pyramid) - 2, -1, -1):
        expanded = cv2.pyrUp(result)
        if expanded.shape != pyramid[i].shape:
            expanded = cv2.resize(expanded, (pyramid[i].shape[1], pyramid[i].shape[0]))
        result = expanded + pyramid[i]
    return result


def laplacian_sharpen(image, levels=4, sharpen_factor=1.5):
    """拉普拉斯金字塔锐化"""
    lap_pyr = build_laplacian_pyramid(image, levels)
    enhanced_pyr = []
    for i, level in enumerate(lap_pyr):
        if i < len(lap_pyr) - 1:
            enhanced_pyr.append(level * sharpen_factor)
        else:
            enhanced_pyr.append(level)
    result = reconstruct_from_laplacian(enhanced_pyr)
    return np.clip(result, 0, 255).astype(np.uint8)


# 创建模糊测试图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.rectangle(image, (30, 30), (226, 226), (100, 150, 200), -1)
cv2.circle(image, (128, 128), 60, (200, 100, 50), -1)
cv2.putText(image, 'SHARP', (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
image = cv2.GaussianBlur(image, (5, 5), 0)

factors = [0.5, 1.0, 1.5, 2.0, 3.0]
results = [laplacian_sharpen(image, sharpen_factor=f) for f in factors]

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('拉普拉斯金字塔锐化', fontsize=14, fontweight='bold')

for i, (factor, result) in enumerate(zip(factors, results)):
    axes[0, i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'Factor = {factor}')
    axes[0, i].axis('off')

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    axes[1, i].imshow(np.abs(edges), cmap='hot')
    axes[1, i].set_title(f'边缘响应 (f={factor})')
    axes[1, i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_laplacian_sharpen.png'), dpi=150, bbox_inches='tight')
plt.show()

print("锐化因子效果说明:")
print("  < 1.0: 平滑（抑制细节）")
print("  = 1.0: 不变")
print("  > 1.0: 锐化（增强细节）")
