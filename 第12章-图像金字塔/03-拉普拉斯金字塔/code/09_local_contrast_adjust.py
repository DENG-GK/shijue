"""
示例9：局部对比度调整
- 高频抑制（去噪）+ 低频增强（对比度）
- 带噪声图像的处理效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def build_laplacian_pyramid(image, levels=4):
    G = [image.astype(np.float64)]
    current = image.astype(np.float64)
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        G.append(current)
    L = []
    for i in range(levels - 1):
        expanded = cv2.pyrUp(G[i + 1])
        if expanded.shape != G[i].shape:
            expanded = cv2.resize(expanded, (G[i].shape[1], G[i].shape[0]))
        L.append(G[i] - expanded)
    L.append(G[-1])
    return L


def reconstruct_from_laplacian(pyramid):
    result = pyramid[-1].copy()
    for i in range(len(pyramid) - 2, -1, -1):
        expanded = cv2.pyrUp(result)
        if expanded.shape != pyramid[i].shape:
            expanded = cv2.resize(expanded, (pyramid[i].shape[1], pyramid[i].shape[0]))
        result = expanded + pyramid[i]
    return result


def local_contrast_adjust(image, levels=4, low_boost=1.5, high_suppress=0.5):
    """局部对比度调整"""
    lap_pyr = build_laplacian_pyramid(image, levels)
    adjusted = []
    for i, level in enumerate(lap_pyr):
        if i == 0:
            adjusted.append(level * high_suppress)  # 抑制高频噪声
        elif i == len(lap_pyr) - 2:
            adjusted.append(level * low_boost)  # 增强低频对比度
        elif i == len(lap_pyr) - 1:
            adjusted.append(level)  # 保留基层
        else:
            adjusted.append(level)
    result = reconstruct_from_laplacian(adjusted)
    return np.clip(result, 0, 255).astype(np.uint8)


# 创建低对比度图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.rectangle(image, (30, 30), (226, 226), (120, 140, 160), -1)
cv2.circle(image, (128, 128), 60, (140, 120, 100), -1)

# 添加噪声
noisy = image.copy()
noise = np.random.normal(0, 15, image.shape).astype(np.int16)
noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# 应用局部对比度调整
adjusted = local_contrast_adjust(noisy, levels=4, low_boost=1.8, high_suppress=0.5)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('局部对比度调整（去噪+增强）', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原图')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('带噪声')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('对比度调整\n（去噪+增强）')
axes[0, 2].axis('off')

diff_noise = cv2.absdiff(image, noisy)
axes[0, 3].imshow(cv2.cvtColor(cv2.convertScaleAbs(diff_noise, alpha=3), cv2.COLOR_BGR2RGB))
axes[0, 3].set_title('噪声(3x)')
axes[0, 3].axis('off')

# 直方图
for i, (img, name) in enumerate([(image, '原图'), (noisy, '噪声'), (adjusted, '调整后')]):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    axes[1, i].hist(gray.flatten(), bins=256, range=[0, 256], alpha=0.7)
    axes[1, i].set_title(f'{name}直方图')
    axes[1, i].set_xlabel('灰度值')

diff_adj = cv2.absdiff(image, adjusted)
axes[1, 3].imshow(cv2.cvtColor(cv2.convertScaleAbs(diff_adj, alpha=3), cv2.COLOR_BGR2RGB))
axes[1, 3].set_title('调整后与原图差异(3x)')
axes[1, 3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_local_contrast_adjust.png'), dpi=150, bbox_inches='tight')
plt.show()

print("局部对比度调整:")
print("  高频抑制 (×0.5): 减少噪声")
print("  低频增强 (×1.8): 提高对比度")
