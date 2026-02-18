"""
示例3：图像缩放方法比较
- INTER_AREA / INTER_LINEAR / INTER_NEAREST / 金字塔
- 缩小到1/8，比较质量
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建高频细节图像
image = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(64):
    for j in range(64):
        if (i + j) % 2 == 0:
            cv2.rectangle(image, (i * 8, j * 8), ((i + 1) * 8, (j + 1) * 8), (200, 200, 200), -1)
cv2.circle(image, (256, 256), 100, (0, 255, 0), -1)
cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)

target = (64, 64)

t0 = time.time()
area = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
t_area = time.time() - t0

t0 = time.time()
linear = cv2.resize(image, target, interpolation=cv2.INTER_LINEAR)
t_linear = time.time() - t0

t0 = time.time()
nearest = cv2.resize(image, target, interpolation=cv2.INTER_NEAREST)
t_nearest = time.time() - t0

t0 = time.time()
pyr = image
for _ in range(3):
    pyr = cv2.pyrDown(pyr)
pyramid_result = cv2.resize(pyr, target, interpolation=cv2.INTER_LINEAR)
t_pyr = time.time() - t0

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('图像缩放方法比较', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f'原图\n{image.shape[1]}×{image.shape[0]}')
axes[0, 0].axis('off')

methods = [('INTER_AREA', area), ('INTER_LINEAR', linear),
           ('INTER_NEAREST', nearest)]
for i, (name, result) in enumerate(methods):
    axes[0, i + 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[0, i + 1].set_title(f'{name}\n{target}')
    axes[0, i + 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(pyramid_result, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f'金字塔\n{target}')
axes[1, 0].axis('off')

# 放大对比
upscale = (256, 256)
for i, (name, result) in enumerate([('AREA', area), ('NEAREST', nearest), ('金字塔', pyramid_result)]):
    up = cv2.resize(result, upscale, interpolation=cv2.INTER_NEAREST)
    axes[1, i + 1].imshow(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
    axes[1, i + 1].set_title(f'{name} (放大)')
    axes[1, i + 1].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_scaling_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"INTER_AREA: {t_area * 1000:.2f} ms")
print(f"INTER_LINEAR: {t_linear * 1000:.2f} ms")
print(f"INTER_NEAREST: {t_nearest * 1000:.2f} ms")
print(f"金字塔: {t_pyr * 1000:.2f} ms")
