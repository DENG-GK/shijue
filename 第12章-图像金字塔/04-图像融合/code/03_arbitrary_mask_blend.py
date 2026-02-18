"""
示例3：任意形状掩码融合
- 圆形/心形/星形掩码
- feather vs no-feather对比
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


size = (256, 256)

img1 = np.zeros((*size, 3), dtype=np.uint8)
img1[:size[0] // 2, :] = [255, 200, 150]
img1[size[0] // 2:, :] = [150, 200, 255]
cv2.circle(img1, (200, 50), 30, (0, 255, 255), -1)

img2 = np.zeros((*size, 3), dtype=np.uint8)
img2[:, :] = [100, 100, 100]
for x in [30, 90, 150, 210]:
    h = np.random.randint(80, 180)
    cv2.rectangle(img2, (x, size[0] - h), (x + 40, size[0]), (70, 70, 70), -1)

# 圆形掩码
circle_mask = np.zeros(size, dtype=np.uint8)
cv2.circle(circle_mask, (128, 128), 80, 255, -1)

# 心形掩码
heart_mask = np.zeros(size, dtype=np.uint8)
pts = []
for t in np.linspace(0, 2 * np.pi, 100):
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts.append((int(128 + x * 4), int(128 + y * 4)))
cv2.fillPoly(heart_mask, [np.array(pts, dtype=np.int32)], 255)

# 星形掩码
star_mask = np.zeros(size, dtype=np.uint8)
center = (128, 128)
star_pts = []
for i in range(5):
    angle = i * 2 * np.pi / 5 - np.pi / 2
    star_pts.append((int(center[0] + 70 * np.cos(angle)), int(center[1] + 70 * np.sin(angle))))
    angle += np.pi / 5
    star_pts.append((int(center[0] + 35 * np.cos(angle)), int(center[1] + 35 * np.sin(angle))))
cv2.fillPoly(star_mask, [np.array(star_pts, dtype=np.int32)], 255)

masks = {'圆形': circle_mask, '心形': heart_mask, '星形': star_mask}

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('任意形状掩码融合', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('图像1')
axes[0, 0].axis('off')

for i, (name, mask) in enumerate(masks.items()):
    axes[0, i + 1].imshow(mask, cmap='gray')
    axes[0, i + 1].set_title(f'{name}掩码')
    axes[0, i + 1].axis('off')

    # 无羽化
    mask_f = mask.astype(np.float64) / 255.0
    no_feather = pyramid_blend(img1, img2, mask_f, levels=6)
    axes[1, i + 1].imshow(cv2.cvtColor(no_feather, cv2.COLOR_BGR2RGB))
    axes[1, i + 1].set_title(f'{name}（无羽化）')
    axes[1, i + 1].axis('off')

    # 有羽化
    mask_feathered = cv2.GaussianBlur(mask_f, (21, 21), 0)
    with_feather = pyramid_blend(img1, img2, mask_feathered, levels=6)
    axes[2, i + 1].imshow(cv2.cvtColor(with_feather, cv2.COLOR_BGR2RGB))
    axes[2, i + 1].set_title(f'{name}（羽化）')
    axes[2, i + 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('图像2')
axes[1, 0].axis('off')
axes[2, 0].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_arbitrary_mask_blend.png'), dpi=150, bbox_inches='tight')
plt.show()
