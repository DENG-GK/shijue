"""
可视化Canny边缘检测的内部处理流程
展示每个步骤的效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_simple_image():
    """创建简单测试图像"""
    img = np.zeros((200, 300), dtype=np.uint8)
    img[:] = 60

    # 矩形
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)

    # 圆形
    cv2.circle(img, (220, 100), 40, 180, -1)

    # 添加少量噪声
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

img = create_simple_image()

print("模拟Canny的处理流程...")

# ===================== 步骤1：高斯滤波 =====================

step1_gaussian = cv2.GaussianBlur(img, (5, 5), 1.4)

# ===================== 步骤2：梯度计算 =====================

sobel_x = cv2.Sobel(step1_gaussian, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(step1_gaussian, cv2.CV_64F, 0, 1, ksize=3)

step2_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
step2_direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

# ===================== 步骤3：非极大值抑制（简化版） =====================

def non_max_suppression(magnitude, direction):
    """简化的非极大值抑制"""
    rows, cols = magnitude.shape
    result = np.zeros_like(magnitude)

    # 将方向量化到4个主方向
    direction = direction % 180

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            angle = direction[i, j]

            # 确定比较的邻居
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            else:
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]

            # 如果是局部最大值，保留
            if magnitude[i, j] >= max(neighbors):
                result[i, j] = magnitude[i, j]

    return result

step3_nms = non_max_suppression(step2_magnitude, step2_direction)

# ===================== 步骤4&5：双阈值和边缘连接 =====================

def double_threshold_and_linking(nms, low_thresh, high_thresh):
    """双阈值检测和边缘连接"""
    rows, cols = nms.shape
    result = np.zeros_like(nms, dtype=np.uint8)

    # 标记强边缘和弱边缘
    strong = 255
    weak = 75

    strong_i, strong_j = np.where(nms >= high_thresh)
    weak_i, weak_j = np.where((nms >= low_thresh) & (nms < high_thresh))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    # 边缘连接：弱边缘与强边缘连通则保留
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if result[i, j] == weak:
                # 检查8邻域是否有强边缘
                if np.any(result[i-1:i+2, j-1:j+2] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0

    return result, (strong_i, strong_j), (weak_i, weak_j)

low_thresh = 50
high_thresh = 150

step4_threshold, strong_pos, weak_pos = double_threshold_and_linking(
    step3_nms, low_thresh, high_thresh)

# 最终结果
step5_final = (step4_threshold == 255).astype(np.uint8) * 255

# OpenCV的Canny结果（对比）
opencv_canny = cv2.Canny(img, low_thresh, high_thresh)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 步骤1
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(step1_gaussian, cmap='gray')
axes[0, 1].set_title('步骤1: 高斯滤波', fontsize=11)
axes[0, 1].axis('off')

# 步骤2
axes[0, 2].imshow(step2_magnitude, cmap='gray', vmax=300)
axes[0, 2].set_title('步骤2: 梯度幅值', fontsize=11)
axes[0, 2].axis('off')

im = axes[0, 3].imshow(step2_direction, cmap='hsv', vmin=-180, vmax=180)
axes[0, 3].set_title('步骤2: 梯度方向', fontsize=11)
axes[0, 3].axis('off')

# 步骤3
axes[1, 0].imshow(step3_nms, cmap='gray', vmax=300)
axes[1, 0].set_title('步骤3: 非极大值抑制', fontsize=11)
axes[1, 0].axis('off')

# 步骤4
axes[1, 1].imshow(step4_threshold, cmap='gray')
axes[1, 1].set_title('步骤4: 双阈值\n白=强边缘，灰=弱边缘', fontsize=11)
axes[1, 1].axis('off')

# 步骤5
axes[1, 2].imshow(step5_final, cmap='gray')
axes[1, 2].set_title('步骤5: 边缘连接\n（简化实现）', fontsize=11)
axes[1, 2].axis('off')

# OpenCV Canny
axes[1, 3].imshow(opencv_canny, cmap='gray')
axes[1, 3].set_title('OpenCV Canny\n（官方实现）', fontsize=11)
axes[1, 3].axis('off')

plt.suptitle('Canny边缘检测处理流程', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('canny_steps.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n处理流程图已保存为 'canny_steps.png'")
