"""
两种自适应方法的对比
ADAPTIVE_THRESH_MEAN_C vs ADAPTIVE_THRESH_GAUSSIAN_C
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_test_doc():
    img = np.ones((300, 400), dtype=np.uint8) * 200

    # 添加文字
    cv2.putText(img, "MEAN vs GAUSSIAN", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 30, 2)
    cv2.putText(img, "Compare adaptive", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 50, 2)
    cv2.putText(img, "threshold methods", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 50, 2)

    # 添加渐变光照
    for j in range(400):
        factor = 0.6 + 0.4 * (j / 400)
        img[:, j] = (img[:, j] * factor).astype(np.uint8)

    # 添加噪声
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

img = create_test_doc()

# ===================== 两种自适应方法 =====================

block_size = 15
C = 5

mean_result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, block_size, C)
gaussian_result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, block_size, C)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

axes[1].imshow(mean_result, cmap='gray')
axes[1].set_title(f'ADAPTIVE_THRESH_MEAN_C\nblockSize={block_size}, C={C}', fontsize=11)
axes[1].axis('off')

axes[2].imshow(gaussian_result, cmap='gray')
axes[2].set_title(f'ADAPTIVE_THRESH_GAUSSIAN_C\nblockSize={block_size}, C={C}', fontsize=11)
axes[2].axis('off')

plt.suptitle('均值法 vs 高斯法', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mean_vs_gaussian.png', dpi=150, bbox_inches='tight')
plt.show()

print("两种方法的区别：")
print("- MEAN_C: 使用简单平均，对噪声更敏感")
print("- GAUSSIAN_C: 使用高斯加权平均，中心像素权重更大，抗噪声能力更强")
print("\n一般推荐使用 GAUSSIAN_C")
