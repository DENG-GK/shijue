"""
示例10：完整的图像融合类
- PyramidBlender类
- blend / blend_multiple / create_gradient_mask
- 多种融合效果演示
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class PyramidBlender:
    """基于拉普拉斯金字塔的图像融合类"""

    def __init__(self, levels=6):
        self.levels = levels

    def _build_gaussian(self, image):
        pyramid = [image.astype(np.float64)]
        current = image.astype(np.float64)
        for i in range(self.levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    def _build_laplacian(self, image):
        G = self._build_gaussian(image)
        L = []
        for i in range(self.levels - 1):
            expanded = cv2.pyrUp(G[i + 1])
            if expanded.shape != G[i].shape:
                expanded = cv2.resize(expanded, (G[i].shape[1], G[i].shape[0]))
            L.append(G[i] - expanded)
        L.append(G[-1])
        return L

    def _reconstruct(self, pyramid):
        result = pyramid[-1].copy()
        for i in range(len(pyramid) - 2, -1, -1):
            expanded = cv2.pyrUp(result)
            if expanded.shape != pyramid[i].shape:
                expanded = cv2.resize(expanded, (pyramid[i].shape[1], pyramid[i].shape[0]))
            result = expanded + pyramid[i]
        return result

    def blend(self, img1, img2, mask):
        """融合两张图像"""
        mask = mask.astype(np.float64)
        if mask.max() > 1:
            mask = mask / 255.0
        L1 = self._build_laplacian(img1)
        L2 = self._build_laplacian(img2)
        GM = self._build_gaussian(mask)
        L_blend = []
        for l1, l2, gm in zip(L1, L2, GM):
            if len(l1.shape) == 3 and len(gm.shape) == 2:
                gm = np.stack([gm] * 3, axis=2)
            L_blend.append(l1 * gm + l2 * (1 - gm))
        return np.clip(self._reconstruct(L_blend), 0, 255).astype(np.uint8)

    def blend_multiple(self, images, masks):
        """融合多张图像"""
        L_pyrs = [self._build_laplacian(img) for img in images]
        G_masks = [self._build_gaussian(m.astype(np.float64)) for m in masks]
        L_blend = []
        for level in range(self.levels):
            blended = np.zeros_like(L_pyrs[0][level])
            for L_pyr, G_mask in zip(L_pyrs, G_masks):
                lap = L_pyr[level]
                w = G_mask[level]
                if len(lap.shape) == 3 and len(w.shape) == 2:
                    w = np.stack([w] * 3, axis=2)
                blended += lap * w
            L_blend.append(blended)
        return np.clip(self._reconstruct(L_blend), 0, 255).astype(np.uint8)

    def create_gradient_mask(self, size, direction='horizontal', position=0.5, width=0.2):
        """创建渐变掩码"""
        h, w = size
        if direction == 'horizontal':
            x = np.linspace(0, 1, w)
            mask = 1 - 1 / (1 + np.exp(-10 * (x - position) / width))
            return np.tile(mask, (h, 1))
        else:
            y = np.linspace(0, 1, h)
            mask = 1 - 1 / (1 + np.exp(-10 * (y - position) / width))
            return np.tile(mask.reshape(-1, 1), (1, w))


# 演示
blender = PyramidBlender(levels=6)
size = (256, 256)

img1 = np.zeros((*size, 3), dtype=np.uint8)
img1[:, :] = [50, 100, 200]
cv2.rectangle(img1, (50, 50), (200, 200), (30, 80, 180), -1)

img2 = np.zeros((*size, 3), dtype=np.uint8)
img2[:, :] = [200, 100, 50]
cv2.circle(img2, (128, 128), 80, (180, 80, 30), -1)

masks = {
    '水平': blender.create_gradient_mask(size, 'horizontal', 0.5, 0.2),
    '垂直': blender.create_gradient_mask(size, 'vertical', 0.5, 0.2),
    '锐边': blender.create_gradient_mask(size, 'horizontal', 0.5, 0.01),
    '平滑': blender.create_gradient_mask(size, 'horizontal', 0.5, 0.5),
}

results = {name: blender.blend(img1, img2, mask) for name, mask in masks.items()}

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('PyramidBlender 类演示', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('图像1')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('图像2')
axes[0, 1].axis('off')

for i, (name, result) in enumerate(results.items()):
    axes[0, 2 + i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[0, 2 + i].set_title(name)
    axes[0, 2 + i].axis('off')

axes[1, 0].axis('off')
axes[1, 1].axis('off')

for i, (name, mask) in enumerate(masks.items()):
    axes[1, 2 + i].imshow(mask, cmap='gray')
    axes[1, 2 + i].set_title(f'{name}掩码')
    axes[1, 2 + i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_pyramid_blender_class.png'), dpi=150, bbox_inches='tight')
plt.show()

print("PyramidBlender 类演示完成！")
print("支持: blend(), blend_multiple(), create_gradient_mask()")
