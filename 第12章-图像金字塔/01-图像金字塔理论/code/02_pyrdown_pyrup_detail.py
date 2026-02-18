"""
示例2：pyrDown和pyrUp的详细使用
- 默认下采样 vs 自定义尺寸
- 多层金字塔构建
- 重建与原图的差异分析
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建棋盘格+圆形测试图像
image = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(8):
    for j in range(8):
        color = ((i + j) % 2) * 200 + 55
        cv2.rectangle(image, (i * 64, j * 64), ((i + 1) * 64, (j + 1) * 64),
                      (color, color // 2, 255 - color), -1)
cv2.circle(image, (256, 256), 100, (0, 255, 255), -1)

# 构建多层金字塔
levels = [image]
current = image
for i in range(4):
    current = cv2.pyrDown(current)
    levels.append(current)
    print(f"Level {i + 1}: {current.shape}")

# 从顶层重建
reconstructed = levels[-1]
for i in range(len(levels) - 1):
    reconstructed = cv2.pyrUp(reconstructed)
    target_size = levels[-(i + 2)].shape[:2][::-1]
    if reconstructed.shape[:2][::-1] != target_size:
        reconstructed = cv2.resize(reconstructed, target_size)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('pyrDown / pyrUp 详细演示', fontsize=14, fontweight='bold')

# 金字塔各层
for i, level in enumerate(levels):
    axes[0, i].imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'Level {i}\n{level.shape[1]}×{level.shape[0]}')
    axes[0, i].axis('off')

# 原图 vs 重建
axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('原图')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('重建图像')
axes[1, 1].axis('off')

if reconstructed.shape == image.shape:
    diff = cv2.absdiff(image, reconstructed)
    diff_amp = cv2.convertScaleAbs(diff, alpha=5)
    axes[1, 2].imshow(cv2.cvtColor(diff_amp, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'差异(5x放大)\n均值: {np.mean(diff):.2f}')
    axes[1, 2].axis('off')

    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    axes[1, 3].hist(gray_diff.flatten(), bins=50, color='steelblue', alpha=0.7)
    axes[1, 3].set_xlabel('差异值')
    axes[1, 3].set_title('差异直方图')
else:
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')

axes[1, 4].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_pyrdown_pyrup_detail.png'), dpi=150, bbox_inches='tight')
plt.show()
