"""
示例8：抗锯齿缩放
- 棋盘格高频图像
- naive resize vs INTER_AREA vs 金字塔
- aliasing评估
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建高频棋盘格
image = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(64):
    for j in range(64):
        if (i + j) % 2 == 0:
            cv2.rectangle(image, (i * 8, j * 8), ((i + 1) * 8, (j + 1) * 8),
                          (255, 255, 255), -1)
cv2.circle(image, (256, 256), 100, (0, 255, 0), 3)
cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), 3)

target = (64, 64)


def pyramid_resize(img, target_size):
    current = img.copy()
    while current.shape[1] > target_size[0] * 2 and current.shape[0] > target_size[1] * 2:
        current = cv2.pyrDown(current)
    return cv2.resize(current, target_size, interpolation=cv2.INTER_LINEAR)


naive = cv2.resize(image, target, interpolation=cv2.INTER_LINEAR)
area = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
pyr_result = pyramid_resize(image, target)


def aliasing_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.std(np.sqrt(gx ** 2 + gy ** 2))


fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('抗锯齿缩放对比', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f'原图\n{image.shape[1]}×{image.shape[0]}')
axes[0, 0].axis('off')

for i, (name, result) in enumerate([('LINEAR', naive), ('AREA', area), ('金字塔', pyr_result)]):
    axes[0, i + 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[0, i + 1].set_title(f'{name}\n{target}')
    axes[0, i + 1].axis('off')

# 放大对比
upscale = (256, 256)
methods = ['LINEAR', 'AREA', '金字塔']
scores = [aliasing_score(naive), aliasing_score(area), aliasing_score(pyr_result)]

for i, (name, result) in enumerate(zip(methods, [naive, area, pyr_result])):
    up = cv2.resize(result, upscale, interpolation=cv2.INTER_NEAREST)
    axes[1, i].imshow(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f'{name} (放大)')
    axes[1, i].axis('off')

bars = axes[1, 3].bar(methods, scores, color=['red', 'orange', 'green'], edgecolor='black')
axes[1, 3].set_ylabel('锯齿分数')
axes[1, 3].set_title('质量对比\n（越低越好）')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_antialiased_resize.png'), dpi=150, bbox_inches='tight')
plt.show()

for name, score in zip(methods, scores):
    print(f"{name}: 锯齿分数 = {score:.2f}")
