"""
示例13：对比度增强方法对比与评估
- 线性拉伸、直方图均衡化、CLAHE、Gamma、Sigmoid
- 多指标评估：对比度、熵、边缘强度、动态范围
- 综合对比与可视化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_enhancement(original, enhanced):
    """多指标评估增强质量"""
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original
        enh_gray = enhanced

    metrics = {}

    # 对比度（标准差）
    metrics['contrast_orig'] = float(np.std(orig_gray))
    metrics['contrast_enh'] = float(np.std(enh_gray))
    metrics['contrast_gain'] = metrics['contrast_enh'] / (metrics['contrast_orig'] + 1e-5)

    # 动态范围
    metrics['range_orig'] = int(np.max(orig_gray)) - int(np.min(orig_gray))
    metrics['range_enh'] = int(np.max(enh_gray)) - int(np.min(enh_gray))

    # 信息熵
    def calc_entropy(img):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    metrics['entropy_orig'] = calc_entropy(orig_gray)
    metrics['entropy_enh'] = calc_entropy(enh_gray)

    # 平均亮度
    metrics['mean_orig'] = float(np.mean(orig_gray))
    metrics['mean_enh'] = float(np.mean(enh_gray))

    # 边缘强度（Sobel）
    def edge_strength(img):
        sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(np.sqrt(sx ** 2 + sy ** 2)))

    metrics['edges_orig'] = edge_strength(orig_gray)
    metrics['edges_enh'] = edge_strength(enh_gray)

    return metrics


# 创建低对比度测试图像
image = np.zeros((300, 400, 3), dtype=np.uint8)
image[:, :200] = [60, 80, 100]
image[:, 200:] = [100, 120, 140]
cv2.circle(image, (100, 150), 60, (80, 100, 120), -1)
cv2.circle(image, (300, 150), 60, (90, 110, 130), -1)
noise = np.random.normal(0, 5, image.shape).astype(np.int16)
image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# ========== 应用各种增强方法 ==========
methods = {}

# 1. 线性拉伸
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
p2, p98 = np.percentile(gray, (2, 98))
linear = np.clip((image.astype(np.float64) - p2) * 255 / (p98 - p2 + 1), 0, 255).astype(np.uint8)
methods['线性拉伸'] = linear

# 2. 直方图均衡化
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l_eq = cv2.equalizeHist(l)
methods['直方图均衡'] = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

# 3. CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)
methods['CLAHE'] = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

# 4. Gamma
gamma = 0.6
table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
methods['Gamma 0.6'] = cv2.LUT(image, table)

# 5. Sigmoid
def sigmoid_enhance(img, gain=8):
    normalized = img / 255.0
    enhanced = 1 / (1 + np.exp(-gain * (normalized - 0.5)))
    return (enhanced * 255).astype(np.uint8)

methods['Sigmoid'] = sigmoid_enhance(image)

# ========== 评估 ==========
results = {}
for name, enhanced in methods.items():
    results[name] = evaluate_enhancement(image, enhanced)

# ========== 可视化 ==========
fig = plt.figure(figsize=(18, 14))
fig.suptitle('对比度增强方法对比与评估', fontsize=14, fontweight='bold')

# 第一行：图像对比
ax_orig = fig.add_subplot(3, 6, 1)
ax_orig.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax_orig.set_title('原图')
ax_orig.axis('off')

for i, (name, enhanced) in enumerate(methods.items()):
    ax = fig.add_subplot(3, 6, i + 2)
    ax.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    ax.set_title(name, fontsize=10)
    ax.axis('off')

# 第二行：指标对比（4个柱状图）
labels = ['原图'] + list(methods.keys())
colors = ['gray'] + ['steelblue'] * len(methods)
first_key = list(methods.keys())[0]

# 对比度
ax1 = fig.add_subplot(3, 4, 5)
vals = [results[first_key]['contrast_orig']] + [results[n]['contrast_enh'] for n in methods]
ax1.bar(range(len(labels)), vals, color=colors)
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('标准差')
ax1.set_title('对比度')

# 信息熵
ax2 = fig.add_subplot(3, 4, 6)
vals = [results[first_key]['entropy_orig']] + [results[n]['entropy_enh'] for n in methods]
ax2.bar(range(len(labels)), vals, color=colors)
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('比特')
ax2.set_title('信息熵')

# 边缘强度
ax3 = fig.add_subplot(3, 4, 7)
vals = [results[first_key]['edges_orig']] + [results[n]['edges_enh'] for n in methods]
ax3.bar(range(len(labels)), vals, color=colors)
ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('平均梯度')
ax3.set_title('边缘强度')

# 动态范围
ax4 = fig.add_subplot(3, 4, 8)
vals = [results[first_key]['range_orig']] + [results[n]['range_enh'] for n in methods]
ax4.bar(range(len(labels)), vals, color=colors)
ax4.set_xticks(range(len(labels)))
ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Max - Min')
ax4.set_title('动态范围')

# 第三行：原图直方图 + 各方法叠加直方图
ax5 = fig.add_subplot(3, 2, 5)
gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ax5.hist(gray_orig.flatten(), bins=256, range=[0, 256], alpha=0.7)
ax5.set_title('原始直方图')
ax5.set_xlabel('灰度值')

ax6 = fig.add_subplot(3, 2, 6)
for name, enhanced in methods.items():
    gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    ax6.hist(gray_enh.flatten(), bins=256, range=[0, 256], alpha=0.4, label=name)
ax6.set_title('各方法直方图叠加')
ax6.set_xlabel('灰度值')
ax6.legend(fontsize=8)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '13_method_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

# 打印详细指标
print("\n" + "=" * 80)
print("对比度增强方法比较")
print("=" * 80)
print(f"\n{'方法':<12} {'对比度':>8} {'信息熵':>8} {'边缘强度':>10} {'动态范围':>10}")
print("-" * 52)
print(f"{'原图':<12} {results[first_key]['contrast_orig']:>8.2f} "
      f"{results[first_key]['entropy_orig']:>8.2f} "
      f"{results[first_key]['edges_orig']:>10.2f} "
      f"{results[first_key]['range_orig']:>10}")
for name in methods:
    r = results[name]
    print(f"{name:<12} {r['contrast_enh']:>8.2f} {r['entropy_enh']:>8.2f} "
          f"{r['edges_enh']:>10.2f} {r['range_enh']:>10}")
print("-" * 52)
