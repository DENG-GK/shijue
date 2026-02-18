"""
示例2：渐变掩码融合
- Sigmoid渐变mask
- Sharp/Medium/Smooth三种过渡宽度
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def build_gaussian_pyramid(image, levels):
    pyramid = [image.astype(np.float64)]
    current = image.astype(np.float64)
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid(image, levels):
    G = build_gaussian_pyramid(image, levels)
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


def pyramid_blend(img1, img2, mask, levels=6):
    L1 = build_laplacian_pyramid(img1, levels)
    L2 = build_laplacian_pyramid(img2, levels)
    GM = build_gaussian_pyramid(mask, levels)
    L_blend = []
    for l1, l2, gm in zip(L1, L2, GM):
        if len(l1.shape) == 3 and len(gm.shape) == 2:
            gm = np.stack([gm] * 3, axis=2)
        L_blend.append(l1 * gm + l2 * (1 - gm))
    return np.clip(reconstruct_from_laplacian(L_blend), 0, 255).astype(np.uint8)


def create_gradient_mask(size, direction='vertical', center=0.5, width=0.3):
    h, w = size
    if direction == 'horizontal':
        x = np.linspace(0, 1, w)
        mask = 1 - 1 / (1 + np.exp(-10 * (x - center) / width))
        mask = np.tile(mask, (h, 1))
    else:
        y = np.linspace(0, 1, h)
        mask = 1 - 1 / (1 + np.exp(-10 * (y - center) / width))
        mask = np.tile(mask.reshape(-1, 1), (1, w))
    return mask


size = (256, 256)

# 天空
img1 = np.zeros((*size, 3), dtype=np.uint8)
for i in range(size[0]):
    img1[i, :] = [255 - i // 2, 200 - i // 3, 100]
cv2.circle(img1, (200, 50), 30, (0, 255, 255), -1)

# 地面
img2 = np.zeros((*size, 3), dtype=np.uint8)
for i in range(size[0]):
    img2[i, :] = [50 + i // 4, 100 + i // 3, 50 + i // 2]
cv2.rectangle(img2, (50, 100), (100, 200), (80, 80, 80), -1)

masks = {
    '锐边': create_gradient_mask(size, 'vertical', 0.5, 0.01),
    '中等': create_gradient_mask(size, 'vertical', 0.5, 0.2),
    '平滑': create_gradient_mask(size, 'vertical', 0.5, 0.5),
}

results = {name: pyramid_blend(img1, img2, mask, levels=5) for name, mask in masks.items()}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('渐变掩码融合', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('图像1（天空）')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('图像2（地面）')
axes[0, 1].axis('off')

for i, (name, mask) in enumerate(masks.items()):
    axes[0, 2 + i if i < 2 else 0].imshow(mask, cmap='gray') if i < 2 else None

for i, (name, mask) in enumerate(masks.items()):
    axes[1, i].imshow(mask, cmap='gray')
    axes[1, i].set_title(f'{name}掩码')
    axes[1, i].axis('off')

axes[1, 3].axis('off')

# 重新布局
fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
fig2.suptitle('不同过渡宽度的融合效果', fontsize=14, fontweight='bold')

for i, (name, mask) in enumerate(masks.items()):
    axes2[0, i].imshow(mask, cmap='gray')
    axes2[0, i].set_title(f'{name}掩码')
    axes2[0, i].axis('off')

for i, (name, result) in enumerate(results.items()):
    axes2[1, i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes2[1, i].set_title(f'{name}融合')
    axes2[1, i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
fig2.savefig(os.path.join(save_dir, '02_gradient_mask_blend.png'), dpi=150, bbox_inches='tight')
plt.show()
