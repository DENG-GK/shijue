"""
示例5：全景图拼接融合
- 左右两张有重叠的图像
- 简单alpha混合 vs 金字塔融合
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


h, w = 200, 300
overlap = 100

left = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(w):
    left[:, i] = [50 + i // 3, 100, 200 - i // 3]
cv2.rectangle(left, (50, 50), (150, 150), (0, 200, 255), -1)
cv2.putText(left, 'LEFT', (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

right = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(w):
    right[:, i] = [min(255, 150 + i // 3), 100, max(0, 100 - i // 5)]
cv2.circle(right, (200, 100), 50, (255, 100, 0), -1)
cv2.putText(right, 'RIGHT', (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# 拼接画布
pw = w * 2 - overlap
left_canvas = np.zeros((h, pw, 3), dtype=np.uint8)
left_canvas[:, :w] = left
right_canvas = np.zeros((h, pw, 3), dtype=np.uint8)
right_canvas[:, w - overlap:] = right

# 掩码
mask = np.zeros((h, pw), dtype=np.float64)
mask[:, :w - overlap] = 1.0
for i in range(overlap):
    mask[:, w - overlap + i] = 1.0 - i / overlap

blended = pyramid_blend(left_canvas, right_canvas, mask, levels=5)

# 简单alpha混合
simple = left_canvas.copy()
simple[:, w - overlap:w] = cv2.addWeighted(
    left_canvas[:, w - overlap:w], 0.5, right_canvas[:, w - overlap:w], 0.5, 0)
simple[:, w:] = right_canvas[:, w:]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('全景图拼接融合', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('左图')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('右图')
axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(simple, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('简单Alpha混合')
axes[1, 0].axis('off')

axes[1, 1].imshow(mask, cmap='gray')
axes[1, 1].set_title('融合掩码')
axes[1, 1].axis('off')

axes[2, 0].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title('金字塔融合（无缝）')
axes[2, 0].axis('off')

diff = cv2.absdiff(simple, blended)
axes[2, 1].imshow(cv2.cvtColor(cv2.convertScaleAbs(diff, alpha=5), cv2.COLOR_BGR2RGB))
axes[2, 1].set_title('两种方法差异(5x)')
axes[2, 1].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_panorama_blend.png'), dpi=150, bbox_inches='tight')
plt.show()
