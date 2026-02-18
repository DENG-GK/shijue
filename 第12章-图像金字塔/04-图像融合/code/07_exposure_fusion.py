"""
示例7：曝光融合
- 模拟欠曝/正常/过曝三张
- 曝光质量权重计算
- 多图金字塔融合
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


size = (256, 256)

# 基础场景
base = np.zeros((*size, 3), dtype=np.uint8)
base[:100, :] = [255, 200, 150]  # 天空
base[100:, :] = [50, 80, 50]  # 地面
cv2.circle(base, (200, 40), 25, (255, 255, 255), -1)  # 太阳
cv2.rectangle(base, (60, 80), (80, 200), (30, 50, 30), -1)  # 树干
cv2.circle(base, (70, 60), 40, (20, 60, 20), -1)  # 树冠
cv2.rectangle(base, (150, 140), (220, 200), (80, 80, 120), -1)  # 房子

under = cv2.convertScaleAbs(base, alpha=0.4, beta=0)
over = np.clip(cv2.convertScaleAbs(base, alpha=2.0, beta=30), 0, 255)
normal = base.copy()


def exposure_weight(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    well = np.exp(-((gray - 128) ** 2) / (2 * 50 ** 2))
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    contrast = lap / (lap.max() + 1e-10)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float64) / 255.0
    return well * (contrast ** 0.5) * (sat ** 0.5)


weights = [exposure_weight(img) for img in [under, normal, over]]
weight_sum = sum(weights) + 1e-10
weights = [w / weight_sum for w in weights]

levels = 5
L_pyrs = [build_laplacian_pyramid(img, levels) for img in [under, normal, over]]
G_ws = [build_gaussian_pyramid(w, levels) for w in weights]

L_fused = []
for level in range(levels):
    fused = np.zeros_like(L_pyrs[0][level])
    for L_pyr, G_w in zip(L_pyrs, G_ws):
        lap = L_pyr[level]
        w = G_w[level]
        if len(lap.shape) == 3 and len(w.shape) == 2:
            w = np.stack([w] * 3, axis=2)
        fused += lap * w
    L_fused.append(fused)

fused = np.clip(reconstruct_from_laplacian(L_fused), 0, 255).astype(np.uint8)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('曝光融合', fontsize=14, fontweight='bold')

for i, (img, name) in enumerate([(under, '欠曝'), (normal, '正常'), (over, '过曝')]):
    axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(name)
    axes[0, i].axis('off')

axes[0, 3].imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title('融合结果')
axes[0, 3].axis('off')

for i, (w, name) in enumerate(zip(weights, ['欠曝', '正常', '过曝'])):
    axes[1, i].imshow(w, cmap='hot')
    axes[1, i].set_title(f'{name}权重')
    axes[1, i].axis('off')

gray_fused = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
axes[1, 3].hist(gray_fused.flatten(), bins=50, color='green', alpha=0.7)
axes[1, 3].set_title('融合直方图')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_exposure_fusion.png'), dpi=150, bbox_inches='tight')
plt.show()
