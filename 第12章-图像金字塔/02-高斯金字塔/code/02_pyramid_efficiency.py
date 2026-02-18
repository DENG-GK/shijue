"""
示例2：金字塔内存和计算分析
- 不同尺寸图像的金字塔效率
- 内存比、存储开销
- 理论极限 4/3 分析
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

test_sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
results = []

for width, height in test_sizes:
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    pyramid = [image]
    current = image
    levels = int(np.log2(min(width, height)))
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)

    original_pixels = width * height
    total_pixels = sum(p.shape[0] * p.shape[1] for p in pyramid)
    memory_ratio = total_pixels / original_pixels

    results.append({
        'size': f'{width}×{height}', 'levels': len(pyramid),
        'original_pixels': original_pixels, 'total_pixels': total_pixels,
        'memory_ratio': memory_ratio, 'overhead': (memory_ratio - 1) * 100
    })

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('高斯金字塔效率分析', fontsize=14, fontweight='bold')

# 内存比
sizes = [r['size'] for r in results]
ratios = [r['memory_ratio'] for r in results]
bars = axes[0].bar(sizes, ratios, color='steelblue', edgecolor='black')
axes[0].axhline(y=4 / 3, color='red', linestyle='--', label='理论极限 (4/3)')
axes[0].set_xlabel('图像尺寸')
axes[0].set_ylabel('内存比')
axes[0].set_title('金字塔内存比')
axes[0].legend()
for bar, ratio in zip(bars, ratios):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{ratio:.3f}', ha='center', va='bottom', fontsize=9)

# 每层像素数（512×512）
image_512 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
pyr_512 = [image_512]
current = image_512
for i in range(8):
    current = cv2.pyrDown(current)
    pyr_512.append(current)

level_pixels = [p.shape[0] * p.shape[1] for p in pyr_512]
axes[1].bar(range(len(pyr_512)), level_pixels, color='coral', edgecolor='black')
axes[1].set_xlabel('金字塔层级')
axes[1].set_ylabel('像素数')
axes[1].set_title('每层像素数 (512×512)')
axes[1].set_yscale('log')

# 累积存储比
cumulative = np.cumsum(level_pixels) / level_pixels[0]
axes[2].plot(range(len(pyr_512)), cumulative, 'go-', linewidth=2, markersize=8)
axes[2].axhline(y=4 / 3, color='red', linestyle='--', label='4/3 极限')
axes[2].set_xlabel('金字塔层级')
axes[2].set_ylabel('累积比')
axes[2].set_title('累积存储比')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_pyramid_efficiency.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print(f"{'尺寸':<12} {'层数':<6} {'原始像素':<12} {'总像素':<12} {'比值':<8} {'开销':<8}")
print("-" * 70)
for r in results:
    print(f"{r['size']:<12} {r['levels']:<6} {r['original_pixels']:<12,} "
          f"{r['total_pixels']:<12,} {r['memory_ratio']:.3f}   {r['overhead']:.1f}%")
print(f"\n理论: 总存储 = 1 + 1/4 + 1/16 + ... = 4/3 ≈ {4/3:.4f}")
