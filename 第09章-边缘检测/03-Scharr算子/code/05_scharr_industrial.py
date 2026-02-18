"""
示例5：Scharr算子的实际应用 - 精密边缘测量
模拟工业视觉中的零件边缘检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建模拟的工业零件图像 =====================

def create_industrial_part():
    """创建一个模拟的工业零件图像"""
    img = np.zeros((400, 500), dtype=np.uint8)
    img[:] = 50  # 暗背景

    # 主体（六边形零件）
    center = (250, 200)
    radius = 120
    pts = []
    for i in range(6):
        angle = i * 60 + 15  # 倾斜15度
        x = int(center[0] + radius * np.cos(np.radians(angle)))
        y = int(center[1] + radius * np.sin(np.radians(angle)))
        pts.append([x, y])
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], 180)

    # 中心孔
    cv2.circle(img, center, 40, 50, -1)

    # 小孔
    for i in range(6):
        angle = i * 60 + 45
        x = int(center[0] + 80 * np.cos(np.radians(angle)))
        y = int(center[1] + 80 * np.sin(np.radians(angle)))
        cv2.circle(img, (x, y), 15, 50, -1)

    # 添加少量噪声（模拟真实拍摄）
    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

img = create_industrial_part()

print("工业零件图像已创建")

# ===================== 边缘检测 =====================

# 预处理
img_blur = cv2.GaussianBlur(img, (3, 3), 0)

# Sobel检测
sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

# Scharr检测
scharr_x = cv2.Scharr(img_blur, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img_blur, cv2.CV_64F, 0, 1)
scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)

# ===================== 亚像素边缘定位 =====================

def find_subpixel_edges(magnitude, threshold=50):
    """简化的亚像素边缘定位"""
    # 二值化找粗略边缘
    _, binary = cv2.threshold(magnitude.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)

    # 细化
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 找轮廓点
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 收集所有边缘点
    edge_points = []
    for contour in contours:
        for point in contour:
            edge_points.append(point[0])

    return np.array(edge_points), binary

sobel_points, sobel_binary = find_subpixel_edges(sobel_mag)
scharr_points, scharr_binary = find_subpixel_edges(scharr_mag)

print(f"\n边缘点数量：")
print(f"  Sobel: {len(sobel_points)} 个点")
print(f"  Scharr: {len(scharr_points)} 个点")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始零件图像', fontsize=12)
axes[0, 0].axis('off')

# Sobel幅值
axes[0, 1].imshow(sobel_mag, cmap='gray', vmax=300)
axes[0, 1].set_title('Sobel梯度幅值', fontsize=12)
axes[0, 1].axis('off')

# Scharr幅值
axes[0, 2].imshow(scharr_mag, cmap='gray', vmax=500)
axes[0, 2].set_title('Scharr梯度幅值', fontsize=12)
axes[0, 2].axis('off')

# Sobel边缘叠加
overlay_sobel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
overlay_sobel[sobel_binary > 0] = [0, 255, 0]
axes[1, 0].imshow(overlay_sobel)
axes[1, 0].set_title('Sobel边缘（绿色）', fontsize=12)
axes[1, 0].axis('off')

# Scharr边缘叠加
overlay_scharr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
overlay_scharr[scharr_binary > 0] = [0, 0, 255]
axes[1, 1].imshow(overlay_scharr)
axes[1, 1].set_title('Scharr边缘（红色）', fontsize=12)
axes[1, 1].axis('off')

# 边缘对比（局部放大）
# 放大六边形的一个角
region = (300, 100, 400, 200)
sobel_crop = sobel_mag[region[1]:region[3], region[0]:region[2]]
scharr_crop = scharr_mag[region[1]:region[3], region[0]:region[2]]

# 差异可视化
diff = np.abs(sobel_crop - scharr_crop)
axes[1, 2].imshow(diff, cmap='hot')
axes[1, 2].set_title('局部差异热力图\n（斜边区域）', fontsize=12)
axes[1, 2].axis('off')

# 在原图上标注放大区域
rect = plt.Rectangle((region[0], region[1]), region[2]-region[0], region[3]-region[1],
                       fill=False, edgecolor='yellow', linewidth=2)
axes[0, 0].add_patch(rect)

plt.suptitle('工业视觉：Scharr精密边缘检测', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('scharr_industrial.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n工业应用图已保存为 'scharr_industrial.png'")
print("\n说明：")
print("  在工业视觉中，精确的边缘检测对于：")
print("  • 尺寸测量")
print("  • 位置定位")
print("  • 缺陷检测")
print("  都非常重要，Scharr的高精度在这些场景中很有价值")
