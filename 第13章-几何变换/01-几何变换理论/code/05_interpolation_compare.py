"""
示例5：插值方法对比
- 小图放大对比NEAREST/LINEAR/CUBIC/LANCZOS4
- 边缘剖面分析
- 3D曲面可视化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建8x8小图
small = np.zeros((8, 8), dtype=np.uint8)
small[2:6, 2:6] = 255
small[3:5, 3:5] = 128

scale = 32
methods = {
    'NEAREST': cv2.INTER_NEAREST,
    'LINEAR': cv2.INTER_LINEAR,
    'CUBIC': cv2.INTER_CUBIC,
    'LANCZOS4': cv2.INTER_LANCZOS4,
}

enlarged = {}
for name, method in methods.items():
    enlarged[name] = cv2.resize(small, None, fx=scale, fy=scale, interpolation=method)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('插值方法详细对比', fontsize=14, fontweight='bold')

# 原始（NEAREST放大显示）
axes[0, 0].imshow(cv2.resize(small, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST),
                   cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title(f'原始 {small.shape[0]}x{small.shape[1]}\n(NEAREST显示)')
axes[0, 0].axis('off')

# 各方法
for i, (name, img) in enumerate(enlarged.items()):
    axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255) if i > 0 else None
    if i > 0:
        axes[0, i].set_title(name)
        axes[0, i].axis('off')

# 修正：单独绘制
for i, (name, img) in enumerate(enlarged.items()):
    axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
    if i == 0:
        axes[0, i].set_title(f'原始 (NEAREST显示)')
    else:
        axes[0, i].set_title(name)
    axes[0, i].axis('off')

# 边缘剖面
axes[1, 0].set_title('水平剖面（中间行）')
row = scale * 4
for name, img in enlarged.items():
    axes[1, 0].plot(img[row, :], label=name, linewidth=1.5)
axes[1, 0].set_xlabel('像素位置')
axes[1, 0].set_ylabel('灰度值')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# 垂直剖面
axes[1, 1].set_title('垂直剖面（中间列）')
col = scale * 4
for name, img in enlarged.items():
    axes[1, 1].plot(img[:, col], label=name, linewidth=1.5)
axes[1, 1].set_xlabel('像素位置')
axes[1, 1].set_ylabel('灰度值')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# 3D曲面: LINEAR
ax3d = fig.add_subplot(2, 4, 7, projection='3d')
step = 8
x = np.arange(enlarged['LINEAR'].shape[1])[::step]
y = np.arange(enlarged['LINEAR'].shape[0])[::step]
X, Y = np.meshgrid(x, y)
Z = enlarged['LINEAR'][::step, ::step]
ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax3d.set_title('LINEAR (3D)')
axes[1, 2].remove()

# 3D曲面: NEAREST
ax3d2 = fig.add_subplot(2, 4, 8, projection='3d')
Z2 = enlarged['NEAREST'][::step, ::step]
ax3d2.plot_surface(X, Y, Z2, cmap='viridis', alpha=0.8)
ax3d2.set_title('NEAREST (3D)')
axes[1, 3].remove()

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_interpolation_compare.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n插值方法特性:")
print(f"{'方法':<12} {'最小值':<8} {'最大值':<8} {'唯一值数':<12}")
print("-" * 42)
for name, img in enlarged.items():
    print(f"{name:<12} {img.min():<8} {img.max():<8} {len(np.unique(img)):<12}")
