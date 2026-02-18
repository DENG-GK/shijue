"""
示例2：图像缩放与插值
- cv2.resize() 用法
- dsize / fx,fy 两种缩放方式
- 5种插值方法对比(NEAREST/LINEAR/AREA/CUBIC/LANCZOS4)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图案
image = np.zeros((100, 100, 3), dtype=np.uint8)
image[:, :] = [220, 220, 220]
cv2.rectangle(image, (20, 20), (80, 80), (0, 255, 0), -1)
cv2.circle(image, (50, 50), 20, (255, 0, 0), -1)
cv2.line(image, (0, 0), (100, 100), (0, 0, 255), 2)
cv2.line(image, (0, 100), (100, 0), (0, 0, 255), 1)

print(f"原始尺寸: {image.shape}")

# 方式1: 使用dsize
resized_dsize = cv2.resize(image, (300, 200))
print(f"dsize缩放: {resized_dsize.shape}")

# 方式2: 使用缩放因子
resized_fx = cv2.resize(image, None, fx=2.5, fy=2.5)
print(f"fx=2.5缩放: {resized_fx.shape}")

# 插值方法对比（放大）
scale = 4
interpolations = {
    'NEAREST': cv2.INTER_NEAREST,
    'LINEAR': cv2.INTER_LINEAR,
    'AREA': cv2.INTER_AREA,
    'CUBIC': cv2.INTER_CUBIC,
    'LANCZOS4': cv2.INTER_LANCZOS4,
}

enlarged = {}
for name, method in interpolations.items():
    enlarged[name] = cv2.resize(image, None, fx=scale, fy=scale, interpolation=method)

# 缩小再放大（查看质量差异）
shrunk = {}
for name, method in interpolations.items():
    small = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=method)
    shrunk[name] = cv2.resize(small, (image.shape[1], image.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

fig, axes = plt.subplots(3, 6, figsize=(20, 10))
fig.suptitle('图像缩放与插值方法', fontsize=14, fontweight='bold')

# 原始图像
axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f'原始\n{image.shape[1]}x{image.shape[0]}')
axes[0, 0].axis('off')

# 放大裁切对比
for i, (name, img) in enumerate(enlarged.items()):
    h, w = img.shape[:2]
    crop = img[h // 4:3 * h // 4, w // 4:3 * w // 4]
    axes[0, i + 1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[0, i + 1].set_title(f'{name}\n(4x放大裁切)')
    axes[0, i + 1].axis('off')

# 缩小再放大对比
axes[1, 0].set_title('缩小→放大')
axes[1, 0].axis('off')
for i, (name, img) in enumerate(shrunk.items()):
    axes[1, i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, i + 1].set_title(f'{name}\n(0.25x→1x)')
    axes[1, i + 1].axis('off')

# 速度测试
import time
axes[2, 0].axis('off')
speed_text = "插值方法推荐:\n\n"
speed_text += "缩小: INTER_AREA\n"
speed_text += "放大: INTER_CUBIC\n"
speed_text += "    或 INTER_LANCZOS4\n"
speed_text += "实时: INTER_LINEAR\n"
speed_text += "快速: INTER_NEAREST"
axes[2, 0].text(0.1, 0.5, speed_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=axes[2, 0].transAxes)

# 速度对比
times = {}
for name, method in interpolations.items():
    start = time.time()
    for _ in range(100):
        cv2.resize(image, None, fx=4, fy=4, interpolation=method)
    times[name] = (time.time() - start) * 10  # ms per call

axes[2, 1].barh(list(times.keys()), list(times.values()), color='steelblue')
axes[2, 1].set_xlabel('时间 (ms)')
axes[2, 1].set_title('放大速度对比 (100次)')

for i in range(2, 6):
    axes[2, i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_resize_interpolation.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n插值方法特性:")
print(f"{'方法':<12} {'最小值':<8} {'最大值':<8} {'唯一值数':<10}")
print("-" * 40)
for name, img in enlarged.items():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"{name:<12} {gray.min():<8} {gray.max():<8} {len(np.unique(gray)):<10}")
