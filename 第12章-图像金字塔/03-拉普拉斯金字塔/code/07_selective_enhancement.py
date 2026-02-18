"""
示例7：选择性细节增强
- level_factors控制每层增强系数
- 细节增强/中频增强/粗特征增强/全增强/细节抑制
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


def selective_enhance(image, levels=4, level_factors=None):
    if level_factors is None:
        level_factors = [1.0] * levels
    lap_pyr = build_laplacian_pyramid(image, levels)
    enhanced = []
    for i, level in enumerate(lap_pyr):
        if i < len(lap_pyr) - 1 and i < len(level_factors):
            enhanced.append(level * level_factors[i])
        else:
            enhanced.append(level)
    result = reconstruct_from_laplacian(enhanced)
    return np.clip(result, 0, 255).astype(np.uint8)


# 创建测试图像（带细节和粗特征）
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.rectangle(image, (30, 30), (226, 226), (100, 150, 200), -1)
cv2.circle(image, (128, 128), 60, (200, 100, 50), -1)
for i in range(20):
    cv2.line(image, (50 + i * 10, 180), (50 + i * 10, 230), (255, 255, 255), 1)
image = cv2.GaussianBlur(image, (3, 3), 0)

strategies = [
    ('原图', [1.0, 1.0, 1.0]),
    ('细节增强', [2.0, 1.0, 1.0]),
    ('中频增强', [1.0, 2.0, 1.0]),
    ('粗特征增强', [1.0, 1.0, 2.0]),
    ('全增强', [1.5, 1.5, 1.5]),
    ('细节抑制', [0.3, 1.0, 1.0]),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('选择性细节增强', fontsize=14, fontweight='bold')

for i, (name, factors) in enumerate(strategies):
    row, col = i // 3, i % 3
    result = selective_enhance(image, levels=4, level_factors=factors)
    axes[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    factors_str = ', '.join([f'{f:.1f}' for f in factors])
    axes[row, col].set_title(f'{name}\n[{factors_str}]')
    axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_selective_enhancement.png'), dpi=150, bbox_inches='tight')
plt.show()

print("选择性增强策略:")
for name, factors in strategies:
    print(f"  {name}: {factors}")
