"""
示例3：在实际图像上对比Sobel和Scharr的效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_complex_image():
    """创建一个复杂的测试图像"""
    img = np.zeros((350, 450), dtype=np.uint8)
    img[:] = 80

    # 建筑物（有各种角度的边缘）
    # 主体
    pts = np.array([[100, 280], [100, 120], [200, 60], [300, 120], [300, 280]], np.int32)
    cv2.fillPoly(img, [pts], 180)

    # 窗户
    cv2.rectangle(img, (130, 150), (170, 200), 100, -1)
    cv2.rectangle(img, (230, 150), (270, 200), 100, -1)

    # 门
    cv2.rectangle(img, (180, 210), (220, 280), 60, -1)

    # 圆形装饰
    cv2.circle(img, (200, 100), 20, 220, -1)

    # 斜线装饰
    cv2.line(img, (50, 300), (120, 250), 200, 3)
    cv2.line(img, (330, 250), (400, 300), 200, 3)

    # 曲线
    for i in range(100):
        x = 350 + i
        y = int(150 + 30 * np.sin(i * 0.1))
        if 0 <= x < 450 and 0 <= y < 350:
            cv2.circle(img, (x, y), 2, 200, -1)

    return img

img = create_complex_image()

# 高斯模糊预处理
img_blur = cv2.GaussianBlur(img, (3, 3), 0)

print("测试图像已创建")

# ===================== Sobel边缘检测 =====================

sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)

# ===================== Scharr边缘检测 =====================

scharr_x = cv2.Scharr(img_blur, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img_blur, cv2.CV_64F, 0, 1)
scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)
scharr_mag = np.clip(scharr_mag, 0, 255).astype(np.uint8)

# ===================== 计算差异 =====================

# 归一化后计算差异
sobel_norm = sobel_mag.astype(np.float64) / max(1, sobel_mag.max())
scharr_norm = scharr_mag.astype(np.float64) / max(1, scharr_mag.max())
diff = np.abs(sobel_norm - scharr_norm)
diff_display = (diff * 255).astype(np.uint8)

print(f"\n差异统计：")
print(f"  最大差异: {diff.max()*100:.1f}%")
print(f"  平均差异: {diff.mean()*100:.1f}%")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

# Sobel结果
axes[0, 1].imshow(sobel_mag, cmap='gray')
axes[0, 1].set_title('Sobel边缘检测', fontsize=12)
axes[0, 1].axis('off')

# Scharr结果
axes[0, 2].imshow(scharr_mag, cmap='gray')
axes[0, 2].set_title('Scharr边缘检测', fontsize=12)
axes[0, 2].axis('off')

# 差异图
im = axes[1, 0].imshow(diff_display, cmap='hot')
axes[1, 0].set_title('差异图（热力图）\n亮色=差异大', fontsize=12)
axes[1, 0].axis('off')
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# 局部放大对比（斜边区域）
region = (50, 150, 200, 300)  # (x1, y1, x2, y2)
sobel_crop = sobel_mag[region[1]:region[3], region[0]:region[2]]
scharr_crop = scharr_mag[region[1]:region[3], region[0]:region[2]]

axes[1, 1].imshow(sobel_crop, cmap='gray')
axes[1, 1].set_title('Sobel局部放大\n（斜边区域）', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(scharr_crop, cmap='gray')
axes[1, 2].set_title('Scharr局部放大\n（斜边区域）', fontsize=12)
axes[1, 2].axis('off')

# 在原图上标注放大区域
rect = plt.Rectangle((region[0], region[1]), region[2]-region[0], region[3]-region[1],
                       fill=False, edgecolor='red', linewidth=2)
axes[0, 0].add_patch(rect)

plt.suptitle('Sobel vs Scharr 实际效果对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_vs_scharr_real.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n对比图已保存为 'sobel_vs_scharr_real.png'")
