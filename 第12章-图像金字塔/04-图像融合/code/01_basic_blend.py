"""
示例1：简单左右融合
- 直接融合 vs 金字塔融合
- 金字塔融合消除可见接缝
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
    result = reconstruct_from_laplacian(L_blend)
    return np.clip(result, 0, 255).astype(np.uint8)


size = (256, 256)

# 图像1：橙色
img1 = np.zeros((*size, 3), dtype=np.uint8)
img1[:, :] = [0, 165, 255]
cv2.circle(img1, (128, 128), 60, (0, 100, 200), -1)

# 图像2：绿色+红色
img2 = np.zeros((*size, 3), dtype=np.uint8)
img2[:, :] = [0, 255, 0]
cv2.circle(img2, (128, 80), 80, (0, 0, 255), -1)

# 掩码
mask = np.zeros(size, dtype=np.float64)
mask[:, :size[1] // 2] = 1.0

# 直接融合
direct = (img1.astype(np.float64) * mask[:, :, np.newaxis] +
          img2.astype(np.float64) * (1 - mask[:, :, np.newaxis])).astype(np.uint8)

# 金字塔融合
blended = pyramid_blend(img1, img2, mask, levels=6)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('图像融合：直接 vs 金字塔', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('图像1')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('图像2')
axes[0, 1].axis('off')

axes[0, 2].imshow(mask, cmap='gray')
axes[0, 2].set_title('掩码')
axes[0, 2].axis('off')

axes[1, 0].imshow(cv2.cvtColor(direct, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('直接融合\n（可见接缝）')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('金字塔融合\n（无缝）')
axes[1, 1].axis('off')

diff = cv2.absdiff(direct, blended)
diff_enhanced = cv2.convertScaleAbs(diff, alpha=5)
axes[1, 2].imshow(cv2.cvtColor(diff_enhanced, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('差异(5x)')
axes[1, 2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_basic_blend.png'), dpi=150, bbox_inches='tight')
plt.show()
