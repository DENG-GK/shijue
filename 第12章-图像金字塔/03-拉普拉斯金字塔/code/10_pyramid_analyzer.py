"""
示例10：拉普拉斯金字塔分析类
- LaplacianPyramidAnalyzer类
- 统计信息、重建、可视化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class LaplacianPyramidAnalyzer:
    """拉普拉斯金字塔综合分析工具"""

    def __init__(self, image, levels=5):
        self.original = image
        self.levels = levels
        self.gaussian_pyr = None
        self.laplacian_pyr = None
        self._build_pyramids()

    def _build_pyramids(self):
        self.gaussian_pyr = [self.original.astype(np.float64)]
        current = self.original.astype(np.float64)
        for i in range(self.levels - 1):
            current = cv2.pyrDown(current)
            self.gaussian_pyr.append(current)

        self.laplacian_pyr = []
        for i in range(self.levels - 1):
            expanded = cv2.pyrUp(self.gaussian_pyr[i + 1])
            if expanded.shape != self.gaussian_pyr[i].shape:
                expanded = cv2.resize(expanded, (self.gaussian_pyr[i].shape[1],
                                                  self.gaussian_pyr[i].shape[0]))
            self.laplacian_pyr.append(self.gaussian_pyr[i] - expanded)
        self.laplacian_pyr.append(self.gaussian_pyr[-1])

    def get_statistics(self):
        stats = []
        for i, level in enumerate(self.laplacian_pyr):
            stats.append({
                'level': i, 'shape': level.shape,
                'mean': np.mean(level), 'std': np.std(level),
                'min': np.min(level), 'max': np.max(level),
                'energy': np.sum(level ** 2)
            })
        return stats

    def reconstruct(self, level_factors=None):
        if level_factors is None:
            level_factors = [1.0] * self.levels
        modified = []
        for i, level in enumerate(self.laplacian_pyr):
            if i < len(level_factors):
                modified.append(level * level_factors[i])
            else:
                modified.append(level)
        result = modified[-1].copy()
        for i in range(len(modified) - 2, -1, -1):
            expanded = cv2.pyrUp(result)
            if expanded.shape != modified[i].shape:
                expanded = cv2.resize(expanded, (modified[i].shape[1], modified[i].shape[0]))
            result = expanded + modified[i]
        return np.clip(result, 0, 255).astype(np.uint8)

    def visualize(self):
        fig, axes = plt.subplots(3, self.levels, figsize=(18, 12))
        fig.suptitle('拉普拉斯金字塔分析', fontsize=14, fontweight='bold')

        # 高斯金字塔
        for i, g in enumerate(self.gaussian_pyr):
            display = np.clip(g, 0, 255).astype(np.uint8)
            if len(display.shape) == 3:
                axes[0, i].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            else:
                axes[0, i].imshow(display, cmap='gray')
            axes[0, i].set_title(f'G{i}: {g.shape[1]}×{g.shape[0]}', fontsize=9)
            axes[0, i].axis('off')

        # 拉普拉斯金字塔
        for i, lap in enumerate(self.laplacian_pyr):
            if i < len(self.laplacian_pyr) - 1:
                display = lap - lap.min()
                if display.max() > 0:
                    display = display / display.max() * 255
                display = display.astype(np.uint8)
            else:
                display = np.clip(lap, 0, 255).astype(np.uint8)
            if len(display.shape) == 3:
                axes[1, i].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            else:
                axes[1, i].imshow(display, cmap='gray')
            axes[1, i].set_title(f'L{i}', fontsize=9)
            axes[1, i].axis('off')

        # 统计信息
        stats = self.get_statistics()
        for i, s in enumerate(stats):
            text = f"均值: {s['mean']:.2f}\n标准差: {s['std']:.2f}\n范围: [{s['min']:.0f}, {s['max']:.0f}]"
            axes[2, i].text(0.5, 0.5, text, ha='center', va='center', fontsize=9,
                            family='monospace', transform=axes[2, i].transAxes)
            axes[2, i].set_title(f'统计 L{i}', fontsize=9)
            axes[2, i].axis('off')

        plt.tight_layout()
        return fig


# 演示
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.rectangle(image, (30, 30), (226, 226), (100, 150, 200), -1)
cv2.circle(image, (128, 128), 60, (200, 100, 50), -1)
for i in range(20):
    cv2.line(image, (50 + i * 10, 180), (50 + i * 10, 230), (255, 255, 255), 1)

analyzer = LaplacianPyramidAnalyzer(image, levels=5)
fig = analyzer.visualize()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_pyramid_analyzer.png'), dpi=150, bbox_inches='tight')
plt.show()

# 测试重建
recon = analyzer.reconstruct()
error = cv2.absdiff(image, recon)
print(f"重建误差: Max={np.max(error)}, Mean={np.mean(error):.4f}")

# 增强重建
recon_sharp = analyzer.reconstruct(level_factors=[2.0, 1.5, 1.0, 1.0, 1.0])
print("增强重建完成（细节×2，中频×1.5）")

stats = analyzer.get_statistics()
print("\n各层统计:")
for s in stats:
    print(f"L{s['level']}: Shape={s['shape'][:2]}, Mean={s['mean']:.2f}, Std={s['std']:.2f}")
