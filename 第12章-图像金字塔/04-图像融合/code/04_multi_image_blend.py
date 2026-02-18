"""
示例4：多图像融合
- 三张图像 + 三个掩码
- 掩码归一化确保和为1
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


def multi_blend(images, masks, levels=6):
    L_pyrs = [build_laplacian_pyramid(img, levels) for img in images]
    G_masks = [build_gaussian_pyramid(m.astype(np.float64), levels) for m in masks]
    L_blend = []
    for level in range(levels):
        blended = np.zeros_like(L_pyrs[0][level])
        for L_pyr, G_mask in zip(L_pyrs, G_masks):
            lap = L_pyr[level]
            weight = G_mask[level]
            if len(lap.shape) == 3 and len(weight.shape) == 2:
                weight = np.stack([weight] * 3, axis=2)
            blended += lap * weight
        L_blend.append(blended)
    return np.clip(reconstruct_from_laplacian(L_blend), 0, 255).astype(np.uint8)


size = (256, 256)
img1 = np.zeros((*size, 3), dtype=np.uint8)
img1[:, :] = [0, 0, 255]
cv2.circle(img1, (64, 128), 40, (0, 0, 200), -1)

img2 = np.zeros((*size, 3), dtype=np.uint8)
img2[:, :] = [0, 255, 0]
cv2.rectangle(img2, (80, 80), (176, 176), (0, 200, 0), -1)

img3 = np.zeros((*size, 3), dtype=np.uint8)
img3[:, :] = [255, 0, 0]
cv2.ellipse(img3, (192, 128), (40, 60), 0, 0, 360, (200, 0, 0), -1)

# 三等分掩码（平滑）
w = size[1]
mask1 = np.zeros(size, dtype=np.float64)
mask1[:, :w // 3] = 1.0
mask2 = np.zeros(size, dtype=np.float64)
mask2[:, w // 3:2 * w // 3] = 1.0
mask3 = np.zeros(size, dtype=np.float64)
mask3[:, 2 * w // 3:] = 1.0

mask1 = cv2.GaussianBlur(mask1, (31, 31), 0)
mask2 = cv2.GaussianBlur(mask2, (31, 31), 0)
mask3 = cv2.GaussianBlur(mask3, (31, 31), 0)

total = mask1 + mask2 + mask3 + 1e-10
mask1 /= total
mask2 /= total
mask3 /= total

blended = multi_blend([img1, img2, img3], [mask1, mask2, mask3], levels=6)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('多图像融合', fontsize=14, fontweight='bold')

for i, (img, name) in enumerate([(img1, '红'), (img2, '绿'), (img3, '蓝')]):
    axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'图像{i + 1} ({name})')
    axes[0, i].axis('off')

axes[0, 3].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title('融合结果')
axes[0, 3].axis('off')

for i, (mask, name) in enumerate([(mask1, '掩码1'), (mask2, '掩码2'), (mask3, '掩码3')]):
    axes[1, i].imshow(mask, cmap='gray')
    axes[1, i].set_title(name)
    axes[1, i].axis('off')

mask_sum = mask1 + mask2 + mask3
axes[1, 3].imshow(mask_sum, cmap='gray')
axes[1, 3].set_title(f'掩码和\n[{mask_sum.min():.3f}, {mask_sum.max():.3f}]')
axes[1, 3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_multi_image_blend.png'), dpi=150, bbox_inches='tight')
plt.show()
