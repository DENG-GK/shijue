"""
示例4：金字塔尺寸计算
- 最大层数：log2(min(W, H))
- 各层尺寸计算
- 存储开销分析（理论极限 4/3）
- 可视化金字塔结构
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def get_pyramid_info(width, height, max_levels=None):
    """计算金字塔各层尺寸"""
    if max_levels is None:
        max_levels = int(np.floor(np.log2(min(width, height))))

    levels = []
    w, h = width, height
    for level in range(max_levels + 1):
        levels.append({
            'level': level,
            'width': w,
            'height': h,
            'pixels': w * h,
            'ratio': (w * h) / (width * height) * 100
        })
        w = (w + 1) // 2
        h = (h + 1) // 2
    return levels


# 测试不同图像尺寸
test_sizes = [(512, 512), (640, 480), (1920, 1080), (1024, 768), (300, 200)]

print("=" * 70)
print("图像金字塔尺寸分析")
print("=" * 70)

for width, height in test_sizes:
    levels = get_pyramid_info(width, height)
    print(f"\n原始尺寸: {width}×{height}, 最大层数: {len(levels) - 1}")
    print(f"{'层级':<6} {'宽度':<8} {'高度':<8} {'像素数':<12} {'占原图%':<10}")
    print("-" * 50)
    for lv in levels[:6]:
        print(f"{lv['level']:<6} {lv['width']:<8} {lv['height']:<8} "
              f"{lv['pixels']:<12} {lv['ratio']:.2f}%")

# 可视化 512×512 金字塔
width, height = 512, 512
levels = get_pyramid_info(width, height)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('金字塔尺寸分析 (512×512)', fontsize=14, fontweight='bold')

# 左图：像素数柱状图
level_nums = [lv['level'] for lv in levels]
pixel_counts = [lv['pixels'] for lv in levels]
percentages = [lv['ratio'] for lv in levels]
colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))

bars = axes[0].bar(level_nums, pixel_counts, color=colors, edgecolor='black')
axes[0].set_xlabel('金字塔层级')
axes[0].set_ylabel('像素数')
axes[0].set_title('每层像素数')
axes[0].set_xticks(level_nums)
for bar, pct in zip(bars, percentages):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=7)

# 右图：累积存储比
cumulative = np.cumsum(pixel_counts) / pixel_counts[0]
axes[1].plot(level_nums, cumulative, 'go-', linewidth=2, markersize=8)
axes[1].axhline(y=4 / 3, color='red', linestyle='--', label='理论极限 (4/3)')
axes[1].set_xlabel('金字塔层级')
axes[1].set_ylabel('累积存储比')
axes[1].set_title('累积存储开销')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_pyramid_sizes.png'), dpi=150, bbox_inches='tight')
plt.show()

total_pixels = sum(pixel_counts)
print(f"\n存储分析：原始{width}×{height}")
print(f"总像素: {total_pixels:,}, 存储比: {total_pixels / pixel_counts[0]:.4f}")
print(f"理论极限: 4/3 ≈ {4 / 3:.4f}")
