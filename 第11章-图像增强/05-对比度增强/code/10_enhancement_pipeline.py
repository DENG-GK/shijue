"""
示例10：综合图像增强流水线
- ImageEnhancer类：自动分析图像特征
- 根据亮度和对比度自动选择增强策略
- 支持暗图像、亮图像、低对比度和正常图像
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ImageEnhancer:
    """综合图像增强流水线"""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def auto_enhance(self, image, method='auto'):
        """自动增强图像"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        mean_val = np.mean(gray)
        std_val = np.std(gray)

        if method == 'auto':
            if mean_val < 80:
                method = 'dark'
            elif mean_val > 180:
                method = 'bright'
            elif std_val < 40:
                method = 'low_contrast'
            else:
                method = 'normal'

        enhance_map = {
            'dark': self._enhance_dark,
            'bright': self._enhance_bright,
            'low_contrast': self._enhance_low_contrast,
            'normal': self._enhance_normal,
        }
        enhanced = enhance_map.get(method, self._enhance_normal)(image)
        return enhanced, method

    def _enhance_dark(self, image):
        """增强暗图像：Gamma + CLAHE"""
        gamma = 0.5
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.LUT(l, table)
            l = self.clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        else:
            result = cv2.LUT(image, table)
            return self.clahe.apply(result)

    def _enhance_bright(self, image):
        """增强过曝图像：提高Gamma"""
        gamma = 1.8
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.LUT(l, table)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        else:
            return cv2.LUT(image, table)

    def _enhance_low_contrast(self, image):
        """增强低对比度图像：直方图均衡化"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        else:
            return cv2.equalizeHist(image)

    def _enhance_normal(self, image):
        """轻度增强正常图像：CLAHE"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        else:
            return self.clahe.apply(image)


# 创建测试图像
base = np.random.randint(100, 200, (300, 400, 3), dtype=np.uint8)
cv2.rectangle(base, (80, 60), (200, 180), (120, 150, 80), -1)
cv2.circle(base, (300, 150), 60, (80, 100, 160), -1)

dark = cv2.convertScaleAbs(base, alpha=0.3, beta=0)
bright = cv2.convertScaleAbs(base, alpha=1.5, beta=50)
low_contrast = cv2.convertScaleAbs(base, alpha=0.3, beta=100)

enhancer = ImageEnhancer()
test_images = [
    (base, '正常图像'),
    (dark, '暗图像'),
    (bright, '亮图像'),
    (low_contrast, '低对比度'),
]

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle('综合图像增强流水线', fontsize=14, fontweight='bold')

for i, (img, name) in enumerate(test_images):
    enhanced, method = enhancer.auto_enhance(img)

    # 原图
    axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f'{name}（原图）')
    axes[i, 0].axis('off')

    # 增强结果
    axes[i, 1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[i, 1].set_title(f'增强（{method}）')
    axes[i, 1].axis('off')

    # 原始直方图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    axes[i, 2].hist(gray.flatten(), bins=256, range=[0, 256], alpha=0.7)
    axes[i, 2].set_title('原始直方图')
    axes[i, 2].set_xlim([0, 256])

    # 增强直方图
    gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    axes[i, 3].hist(gray_enh.flatten(), bins=256, range=[0, 256], alpha=0.7, color='orange')
    axes[i, 3].set_title('增强直方图')
    axes[i, 3].set_xlim([0, 256])

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_enhancement_pipeline.png'), dpi=150, bbox_inches='tight')
plt.show()

print("自动增强流水线：")
print("- 暗图像 → Gamma校正 + CLAHE")
print("- 亮图像 → 反向Gamma校正")
print("- 低对比度 → 直方图均衡化")
print("- 正常图像 → 轻度CLAHE")
