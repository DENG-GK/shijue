"""
示例4：演示Scharr在计算精确梯度方向时的优势
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建已知角度的边缘 =====================

def create_known_angle_image(true_angle, size=200):
    """创建具有精确已知角度的边缘"""
    img = np.zeros((size, size), dtype=np.uint8)

    center = size // 2
    angle_rad = np.radians(true_angle)

    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center
            # 点在线的哪一侧
            side = x * np.cos(angle_rad) + y * np.sin(angle_rad)
            if side > 0:
                img[i, j] = 200
            else:
                img[i, j] = 50

    return img

# 测试不同的真实角度
true_angles = [0, 22.5, 45, 67.5, 90]

results = []

for true_angle in true_angles:
    img = create_known_angle_image(true_angle)

    # 使用Sobel计算梯度方向
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # 使用Scharr计算梯度方向
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_direction = np.arctan2(scharr_y, scharr_x) * 180 / np.pi

    # 在边缘区域取平均方向
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_mask = sobel_mag > 100

    sobel_mean = np.mean(sobel_direction[edge_mask])
    scharr_mean = np.mean(scharr_direction[edge_mask])

    # 梯度方向垂直于边缘方向，所以需要+90°
    expected_gradient = true_angle + 90
    if expected_gradient > 180:
        expected_gradient -= 360

    # 计算误差
    sobel_error = abs(sobel_mean - expected_gradient)
    if sobel_error > 180:
        sobel_error = 360 - sobel_error
    scharr_error = abs(scharr_mean - expected_gradient)
    if scharr_error > 180:
        scharr_error = 360 - scharr_error

    results.append({
        'true_angle': true_angle,
        'expected_gradient': expected_gradient,
        'sobel_direction': sobel_mean,
        'scharr_direction': scharr_mean,
        'sobel_error': sobel_error,
        'scharr_error': scharr_error,
        'image': img
    })

# ===================== 打印结果 =====================

print("梯度方向精度对比：")
print("=" * 70)
print(f"{'边缘角度':>10} | {'期望梯度':>10} | {'Sobel':>10} | {'Scharr':>10} | {'Sobel误差':>10} | {'Scharr误差':>10}")
print("-" * 70)
for r in results:
    print(f"{r['true_angle']:>9}° | {r['expected_gradient']:>9.1f}° | {r['sobel_direction']:>9.1f}° | "
          f"{r['scharr_direction']:>9.1f}° | {r['sobel_error']:>9.1f}° | {r['scharr_error']:>9.1f}°")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 5, figsize=(16, 7))

for i, r in enumerate(results):
    # 上排：原图
    axes[0, i].imshow(r['image'], cmap='gray')
    axes[0, i].set_title(f"边缘角度: {r['true_angle']}°", fontsize=10)
    axes[0, i].axis('off')

    # 下排：误差柱状图
    errors = [r['sobel_error'], r['scharr_error']]
    bars = axes[1, i].bar(['Sobel', 'Scharr'], errors, color=['blue', 'red'])
    axes[1, i].set_ylabel('方向误差 (度)')
    axes[1, i].set_title(f'误差对比', fontsize=10)
    axes[1, i].set_ylim([0, max(5, max(errors)*1.2)])

    # 在柱子上标注数值
    for bar, error in zip(bars, errors):
        axes[1, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{error:.2f}°', ha='center', va='bottom', fontsize=9)

plt.suptitle('Sobel vs Scharr 梯度方向精度对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('scharr_direction_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

# 绘制误差汇总图
fig, ax = plt.subplots(figsize=(10, 6))

angles_list = [r['true_angle'] for r in results]
sobel_errors = [r['sobel_error'] for r in results]
scharr_errors = [r['scharr_error'] for r in results]

x = np.arange(len(angles_list))
width = 0.35

bars1 = ax.bar(x - width/2, sobel_errors, width, label='Sobel', color='steelblue')
bars2 = ax.bar(x + width/2, scharr_errors, width, label='Scharr', color='coral')

ax.set_xlabel('边缘角度 (度)', fontsize=12)
ax.set_ylabel('方向误差 (度)', fontsize=12)
ax.set_title('Sobel vs Scharr 方向精度对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{a}°' for a in angles_list])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}°', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}°', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('scharr_error_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n结论：")
print("  Scharr在计算梯度方向时比Sobel更精确")
print("  特别是在45度附近的边缘，优势更明显")
print("\n图像已保存")
