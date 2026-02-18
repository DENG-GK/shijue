"""
示例12：批量图像对比度增强
- BatchContrastEnhancer类
- 支持clahe/gamma/linear/auto四种方法
- 批量处理目录中的图像
- 生成增强前后对比图
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class BatchContrastEnhancer:
    """批量对比度增强器"""

    def __init__(self, output_dir='enhanced_images'):
        self.output_dir = output_dir
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        os.makedirs(output_dir, exist_ok=True)

    def enhance_single(self, image, method='clahe'):
        """增强单张图像"""
        method_map = {
            'clahe': self._apply_clahe,
            'gamma': self._apply_gamma,
            'linear': self._apply_linear,
            'auto': self._apply_auto,
        }
        return method_map.get(method, self._apply_clahe)(image)

    def _apply_clahe(self, image):
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return self.clahe.apply(image)

    def _apply_gamma(self, image, gamma=0.7):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)

    def _apply_linear(self, image):
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                ch = image[:, :, i]
                p2, p98 = np.percentile(ch, (2, 98))
                result[:, :, i] = np.clip((ch - p2) * 255 / (p98 - p2 + 1), 0, 255)
            return result.astype(np.uint8)
        p2, p98 = np.percentile(image, (2, 98))
        return np.clip((image - p2) * 255 / (p98 - p2 + 1), 0, 255).astype(np.uint8)

    def _apply_auto(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_val = np.mean(gray)
        if mean_val < 80:
            enhanced = self._apply_gamma(image, gamma=0.5)
            return self._apply_clahe(enhanced)
        elif mean_val > 180:
            return self._apply_gamma(image, gamma=1.8)
        return self._apply_clahe(image)

    def process_batch(self, images_dict, method='auto'):
        """批量处理图像字典 {名称: 图像}"""
        stats = {'processed': 0, 'results': {}}
        for name, img in images_dict.items():
            enhanced = self.enhance_single(img, method)
            stats['results'][name] = enhanced
            stats['processed'] += 1
        return stats


# 创建模拟批量图像
sample_images = {}
for i, (low, high, label) in enumerate([
    (20, 80, 'dark'),
    (180, 250, 'bright'),
    (100, 140, 'low_contrast'),
    (30, 230, 'normal'),
]):
    img = np.random.randint(low, high, (200, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 40), (150, 120), np.array([low, high, (low + high) // 2], dtype=np.uint8).tolist(), -1)
    cv2.circle(img, (220, 100), 40, np.array([high, low, (low + high) // 2], dtype=np.uint8).tolist(), -1)
    sample_images[label] = img

# 批量增强
enhancer = BatchContrastEnhancer()
methods = ['clahe', 'gamma', 'linear', 'auto']

fig, axes = plt.subplots(len(sample_images), len(methods) + 1, figsize=(18, 12))
fig.suptitle('批量图像对比度增强', fontsize=14, fontweight='bold')

for row, (name, img) in enumerate(sample_images.items()):
    # 原图
    axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[row, 0].set_title(f'原图 ({name})')
    axes[row, 0].axis('off')

    # 各方法增强结果
    for col, method in enumerate(methods):
        enhanced = enhancer.enhance_single(img, method)
        axes[row, col + 1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        axes[row, col + 1].set_title(method.upper())
        axes[row, col + 1].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '12_batch_enhancement.png'), dpi=150, bbox_inches='tight')
plt.show()

# 统计输出
print("批量增强完成！")
for name, img in sample_images.items():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  {name}: 均值={np.mean(gray):.1f}, 标准差={np.std(gray):.1f}")
