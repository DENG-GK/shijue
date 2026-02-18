"""
示例6：人脸换脸效果
- 两张模拟面部图像
- 椭圆掩码 + 高斯羽化
- 金字塔融合 vs 直接替换
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

# 面部1
face1 = np.ones((*size, 3), dtype=np.uint8) * 200
cv2.ellipse(face1, (128, 140), (80, 100), 0, 0, 360, (180, 160, 140), -1)
cv2.ellipse(face1, (95, 110), (15, 10), 0, 0, 360, (255, 255, 255), -1)
cv2.ellipse(face1, (161, 110), (15, 10), 0, 0, 360, (255, 255, 255), -1)
cv2.circle(face1, (95, 110), 5, (50, 30, 20), -1)
cv2.circle(face1, (161, 110), 5, (50, 30, 20), -1)
cv2.ellipse(face1, (128, 180), (25, 10), 0, 0, 180, (100, 80, 150), -1)
cv2.ellipse(face1, (128, 60), (90, 50), 0, 180, 360, (50, 30, 20), -1)

# 面部2
face2 = np.ones((*size, 3), dtype=np.uint8) * 220
cv2.ellipse(face2, (128, 140), (70, 90), 0, 0, 360, (200, 180, 160), -1)
cv2.ellipse(face2, (100, 115), (12, 8), 0, 0, 360, (255, 255, 255), -1)
cv2.ellipse(face2, (156, 115), (12, 8), 0, 0, 360, (255, 255, 255), -1)
cv2.circle(face2, (100, 115), 4, (80, 50, 30), -1)
cv2.circle(face2, (156, 115), 4, (80, 50, 30), -1)
cv2.ellipse(face2, (128, 175), (20, 8), 0, 0, 180, (150, 100, 120), -1)
pts = np.array([[60, 70], [128, 30], [196, 70], [180, 60], [128, 45], [76, 60]], np.int32)
cv2.fillPoly(face2, [pts], (30, 20, 10))

# 面部掩码
mask = np.zeros(size, dtype=np.float64)
cv2.ellipse(mask, (128, 140), (70, 90), 0, 0, 360, 1.0, -1)
mask = cv2.GaussianBlur(mask, (31, 31), 0)

swapped = pyramid_blend(face1, face2, mask, levels=5)

# 直接替换
direct = face2.copy()
direct[mask > 0.5] = face1[mask > 0.5]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('人脸换脸效果', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('面部1')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('面部2')
axes[0, 1].axis('off')

axes[0, 2].imshow(mask, cmap='gray')
axes[0, 2].set_title('面部掩码')
axes[0, 2].axis('off')

axes[1, 0].imshow(cv2.cvtColor(direct, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('直接替换\n（有伪影）')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('金字塔融合\n（自然过渡）')
axes[1, 1].axis('off')

comparison = np.hstack([face1, face2, swapped])
axes[1, 2].imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('对比')
axes[1, 2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_face_swap.png'), dpi=150, bbox_inches='tight')
plt.show()
