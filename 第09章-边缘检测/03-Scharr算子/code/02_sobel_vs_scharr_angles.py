"""
示例2：对比Sobel和Scharr在不同角度边缘上的检测精度
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建不同角度的边缘 =====================

def create_angled_edge(angle_deg, size=100):
    """创建指定角度的边缘图像"""
    img = np.zeros((size, size), dtype=np.uint8)

    # 使用旋转矩阵创建斜线
    center = size // 2
    angle_rad = np.radians(angle_deg)

    # 创建一条通过中心的线
    for i in range(size):
        for j in range(size):
            # 计算点到线的距离
            x = j - center
            y = i - center
            # 旋转坐标
            x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)
            if x_rot > 0:
                img[i, j] = 200
            else:
                img[i, j] = 50

    return img

# 创建不同角度的边缘图像
angles = [0, 15, 30, 45, 60, 75, 90]
edge_images = {angle: create_angled_edge(angle) for angle in angles}

print("创建了不同角度的边缘图像：", angles)

# ===================== 计算响应 =====================

def compute_max_gradient(img, method='sobel'):
    """计算最大梯度幅值"""
    if method == 'sobel':
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    else:  # scharr
        gx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(img, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(gx**2 + gy**2)

    # 返回中心区域的最大值（避免边界效应）
    center_region = magnitude[30:70, 30:70]
    return np.max(center_region)

# 计算不同角度的响应
sobel_responses = []
scharr_responses = []

for angle in angles:
    img = edge_images[angle]
    sobel_responses.append(compute_max_gradient(img, 'sobel'))
    scharr_responses.append(compute_max_gradient(img, 'scharr'))

# 归一化（以0度为基准）
sobel_norm = [r / sobel_responses[0] for r in sobel_responses]
scharr_norm = [r / scharr_responses[0] for r in scharr_responses]

print("\n归一化响应对比（以0度为100%）：")
print(f"{'角度':>6} | {'Sobel':>8} | {'Scharr':>8} | {'理想':>8}")
print("-" * 40)
for i, angle in enumerate(angles):
    print(f"{angle:>5}° | {sobel_norm[i]*100:>7.1f}% | {scharr_norm[i]*100:>7.1f}% | {100.0:>7.1f}%")

# ===================== 可视化 =====================

fig = plt.figure(figsize=(16, 10))

# 上半部分：不同角度的边缘图像
for i, angle in enumerate(angles):
    ax = plt.subplot(3, 7, i+1)
    ax.imshow(edge_images[angle], cmap='gray')
    ax.set_title(f'{angle}°', fontsize=10)
    ax.axis('off')

# 中间：Sobel检测结果
for i, angle in enumerate(angles):
    ax = plt.subplot(3, 7, 7+i+1)
    img = edge_images[angle]
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    ax.imshow(mag, cmap='gray')
    ax.set_title(f'Sobel\n{sobel_norm[i]*100:.1f}%', fontsize=9)
    ax.axis('off')

# 下半部分：Scharr检测结果
for i, angle in enumerate(angles):
    ax = plt.subplot(3, 7, 14+i+1)
    img = edge_images[angle]
    gx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    ax.imshow(mag, cmap='gray')
    ax.set_title(f'Scharr\n{scharr_norm[i]*100:.1f}%', fontsize=9)
    ax.axis('off')

plt.suptitle('Sobel vs Scharr 不同角度响应对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_vs_scharr_angles.png', dpi=150, bbox_inches='tight')
plt.show()

# 绘制响应曲线
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(angles, [s*100 for s in sobel_norm], 'b-o', linewidth=2, markersize=8, label='Sobel')
ax.plot(angles, [s*100 for s in scharr_norm], 'r-s', linewidth=2, markersize=8, label='Scharr')
ax.axhline(y=100, color='g', linestyle='--', linewidth=2, label='理想响应')

ax.set_xlabel('边缘角度 (度)', fontsize=12)
ax.set_ylabel('归一化响应 (%)', fontsize=12)
ax.set_title('Sobel vs Scharr 角度响应曲线', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(angles)
ax.set_ylim([60, 110])

# 标注45度处的差异
ax.annotate(f'Sobel在45°: {sobel_norm[3]*100:.1f}%\nScharr在45°: {scharr_norm[3]*100:.1f}%',
            xy=(45, sobel_norm[3]*100), xytext=(55, 75),
            fontsize=10, fontfamily='SimHei',
            arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('sobel_vs_scharr_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n对比图已保存")
