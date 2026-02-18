"""
演示Canny算法中阈值参数对结果的影响
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_complex_image():
    """创建复杂测试图像"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 渐变背景
    for x in range(400):
        img[:, x] = 60 + int(x * 0.2)

    # 形状
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (280, 100), 50, 220, -1)

    # 细线条
    cv2.line(img, (50, 200), (200, 280), 180, 2)
    cv2.line(img, (220, 200), (370, 280), 180, 2)

    # 低对比度形状
    cv2.rectangle(img, (250, 180), (350, 250), 130, -1)

    # 添加噪声
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

img = create_complex_image()

# 高斯模糊
img_blur = cv2.GaussianBlur(img, (5, 5), 1.0)

print("测试图像已创建")

# ===================== 不同阈值的效果 =====================

threshold_pairs = [
    (10, 30),    # 非常低
    (30, 90),    # 低
    (50, 150),   # 中等（推荐）
    (100, 200),  # 高
    (150, 300),  # 非常高
]

results = {}
for t1, t2 in threshold_pairs:
    edges = cv2.Canny(img_blur, t1, t2)
    results[(t1, t2)] = edges
    edge_count = np.sum(edges > 0)
    print(f"阈值 ({t1:3d}, {t2:3d}): {edge_count:6d} 个边缘像素")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像（带噪声）', fontsize=11)
axes[0, 0].axis('off')

# 不同阈值
for i, (t1, t2) in enumerate(threshold_pairs):
    row = (i + 1) // 3
    col = (i + 1) % 3
    axes[row, col].imshow(results[(t1, t2)], cmap='gray')

    if (t1, t2) == (50, 150):
        title = f'阈值 ({t1}, {t2})\n✓ 推荐'
        color = 'green'
    elif t1 < 30:
        title = f'阈值 ({t1}, {t2})\n噪声较多'
        color = 'orange'
    elif t1 > 100:
        title = f'阈值 ({t1}, {t2})\n边缘丢失'
        color = 'red'
    else:
        title = f'阈值 ({t1}, {t2})'
        color = 'black'

    axes[row, col].set_title(title, fontsize=11, color=color)
    axes[row, col].axis('off')

plt.suptitle('Canny阈值参数的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('canny_thresholds.png', dpi=150, bbox_inches='tight')
plt.show()

# 绘制边缘像素数量变化
fig, ax = plt.subplots(figsize=(10, 5))

thresholds = [f"({t1},{t2})" for t1, t2 in threshold_pairs]
counts = [np.sum(results[(t1, t2)] > 0) for t1, t2 in threshold_pairs]

bars = ax.bar(thresholds, counts, color=['orange', 'yellow', 'green', 'yellow', 'red'])
ax.set_xlabel('阈值 (低, 高)', fontsize=12)
ax.set_ylabel('边缘像素数量', fontsize=12)
ax.set_title('不同阈值下的边缘像素数量', fontsize=14, fontweight='bold')

# 标注推荐阈值
bars[2].set_color('green')
ax.annotate('推荐', xy=(2, counts[2]), xytext=(2.3, counts[2]*1.1),
            fontsize=12, fontfamily='SimHei',
            arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig('canny_threshold_counts.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n对比图已保存")
