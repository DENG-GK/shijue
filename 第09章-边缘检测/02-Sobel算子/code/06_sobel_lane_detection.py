"""
示例6：Sobel边缘检测的实际应用 - 简化的车道线检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建模拟的道路图像 =====================

def create_road_image():
    """创建一个模拟的道路图像"""
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # 天空（上半部分）
    img[:200, :] = [180, 200, 220]

    # 道路（下半部分）
    for y in range(200, 400):
        img[y, :] = [80, 80, 80]

    # 车道线（白色虚线）
    # 左车道线
    for i in range(0, 200, 40):
        y1 = 200 + i
        y2 = min(200 + i + 25, 399)
        x1 = 200 - i
        x2 = 200 - i - 15
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)

    # 右车道线
    for i in range(0, 200, 40):
        y1 = 200 + i
        y2 = min(200 + i + 25, 399)
        x1 = 400 + i
        x2 = 400 + i + 15
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)

    # 中心线（黄色实线）
    cv2.line(img, (300, 200), (300, 400), (0, 200, 200), 3)

    # 添加一些噪声
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

road_img = create_road_image()

print("模拟道路图像已创建")

# ===================== 车道线检测流程 =====================

# 1. 转换为灰度图
gray = cv2.cvtColor(road_img, cv2.COLOR_BGR2GRAY)

# 2. ROI（感兴趣区域）- 只关注道路部分
roi = gray[200:, :]  # 下半部分

# 3. 高斯模糊
blurred = cv2.GaussianBlur(roi, (5, 5), 0)

# 4. Sobel边缘检测
# 车道线主要是垂直方向的，所以重点检测X方向梯度
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_x_abs = np.abs(sobel_x)
sobel_x_abs = np.clip(sobel_x_abs, 0, 255).astype(np.uint8)

# 也检测Y方向（用于完整性）
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_y_abs = np.abs(sobel_y)
sobel_y_abs = np.clip(sobel_y_abs, 0, 255).astype(np.uint8)

# 合并
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

# 5. 二值化
_, binary = cv2.threshold(sobel_x_abs, 80, 255, cv2.THRESH_BINARY)

# 6. 形态学处理
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

# 7. 将检测结果放回原图大小
result_full = np.zeros_like(gray)
result_full[200:, :] = cleaned

# 8. 标记检测到的车道线
result_color = road_img.copy()
result_color[result_full > 0] = [0, 255, 0]  # 绿色标记

print("车道线检测完成")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 第一行
axes[0, 0].imshow(cv2.cvtColor(road_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('1. 原始道路图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(roi, cmap='gray')
axes[0, 1].set_title('2. ROI区域（道路部分）', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(blurred, cmap='gray')
axes[0, 2].set_title('3. 高斯模糊', fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(sobel_x_abs, cmap='gray')
axes[0, 3].set_title('4. Sobel X（检测垂直线）', fontsize=11)
axes[0, 3].axis('off')

# 第二行
axes[1, 0].imshow(magnitude, cmap='gray')
axes[1, 0].set_title('5. 梯度幅值', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(binary, cmap='gray')
axes[1, 1].set_title('6. 二值化', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(cleaned, cmap='gray')
axes[1, 2].set_title('7. 形态学清理', fontsize=11)
axes[1, 2].axis('off')

axes[1, 3].imshow(cv2.cvtColor(result_color, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title('8. 检测结果（绿色标记）', fontsize=11)
axes[1, 3].axis('off')

plt.suptitle('Sobel边缘检测应用：车道线检测', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_lane_detection.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n车道线检测结果已保存为 'sobel_lane_detection.png'")
print("\n说明：")
print("  这是一个简化的车道线检测示例")
print("  实际应用中还需要：")
print("  - 透视变换（鸟瞰图）")
print("  - 霍夫变换（直线检测）")
print("  - 曲线拟合")
print("  - 时序滤波等")
