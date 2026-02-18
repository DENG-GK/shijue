"""
示例10：直方图比较与相似度计算
- 四种直方图比较方法：相关性、卡方、交集、巴氏距离
- 对比相似图像与不同图像的匹配得分
- 理解各方法的评判标准
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def compare_histograms(img1, img2):
    """比较两幅图像的直方图"""
    # 转换为灰度图
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    # 计算直方图
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # 归一化
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # 四种比较方法
    methods = {
        '相关性 (Correlation)': cv2.HISTCMP_CORREL,           # 越大越相似
        '卡方 (Chi-Square)': cv2.HISTCMP_CHISQR,              # 越小越相似
        '交集 (Intersection)': cv2.HISTCMP_INTERSECT,         # 越大越相似
        '巴氏距离 (Bhattacharyya)': cv2.HISTCMP_BHATTACHARYYA,  # 越小越相似
    }

    results = {}
    for name, method in methods.items():
        score = cv2.compareHist(hist1, hist2, method)
        results[name] = score

    return hist1, hist2, results


# 创建测试图像
img_original = np.random.normal(128, 40, (200, 300)).clip(0, 255).astype(np.uint8)
img_similar = img_original + np.random.normal(0, 10, img_original.shape)
img_similar = img_similar.clip(0, 255).astype(np.uint8)
img_different = np.random.normal(80, 20, (200, 300)).clip(0, 255).astype(np.uint8)

# 比较
hist_orig, hist_sim, results_sim = compare_histograms(img_original, img_similar)
hist_orig2, hist_diff, results_diff = compare_histograms(img_original, img_different)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('直方图比较与相似度计算', fontsize=16, fontweight='bold')

# 图像
axes[0, 0].imshow(img_original, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(img_similar, cmap='gray')
axes[0, 1].set_title('相似图像', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(img_different, cmap='gray')
axes[0, 2].set_title('不同图像', fontsize=11)
axes[0, 2].axis('off')

# 直方图
axes[1, 0].plot(hist_orig, label='原始')
axes[1, 0].set_title('原始图像直方图', fontsize=11)
axes[1, 0].set_xlim([0, 255])
axes[1, 0].set_xlabel('像素值')

axes[1, 1].plot(hist_orig, label='原始', alpha=0.7)
axes[1, 1].plot(hist_sim, label='相似', alpha=0.7)
axes[1, 1].set_title('原始 vs 相似', fontsize=11)
axes[1, 1].set_xlim([0, 255])
axes[1, 1].set_xlabel('像素值')
axes[1, 1].legend()

axes[1, 2].plot(hist_orig2, label='原始', alpha=0.7)
axes[1, 2].plot(hist_diff, label='不同', alpha=0.7)
axes[1, 2].set_title('原始 vs 不同', fontsize=11)
axes[1, 2].set_xlim([0, 255])
axes[1, 2].set_xlabel('像素值')
axes[1, 2].legend()

plt.tight_layout()

# 保存图片
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_histogram_compare.png'), dpi=150, bbox_inches='tight')
plt.show()

# 打印比较结果
print("直方图比较结果：")
print("=" * 70)
print(f"{'方法':<30s} {'相似图像':>15s} {'不同图像':>15s}")
print("=" * 70)
for method in results_sim.keys():
    print(f"{method:<30s} {results_sim[method]:>15.4f} {results_diff[method]:>15.4f}")
print("=" * 70)
print("\n说明：")
print("- Correlation: 相关性，范围[-1, 1]，越大越相似")
print("- Chi-Square: 卡方距离，越小越相似")
print("- Intersection: 交集，越大越相似")
print("- Bhattacharyya: 巴氏距离，范围[0, 1]，越小越相似")
