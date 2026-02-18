"""
示例5：Sobel边缘检测的完整处理流程
包括预处理、边缘检测、后处理
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建或读取图像 =====================

def create_sample_image():
    """创建一个模拟的实际场景图像"""
    img = np.zeros((350, 450, 3), dtype=np.uint8)

    # 背景渐变
    for y in range(350):
        img[y, :] = [100 + y//5, 100 + y//5, 100 + y//5]

    # 建筑物
    cv2.rectangle(img, (50, 100), (180, 300), (80, 80, 120), -1)
    cv2.rectangle(img, (70, 120), (100, 180), (150, 150, 100), -1)  # 窗户
    cv2.rectangle(img, (130, 120), (160, 180), (150, 150, 100), -1)  # 窗户
    cv2.rectangle(img, (100, 220), (140, 300), (60, 60, 80), -1)  # 门

    # 树
    cv2.rectangle(img, (240, 200), (270, 300), (50, 80, 60), -1)  # 树干
    cv2.circle(img, (255, 150), 60, (40, 100, 50), -1)  # 树冠

    # 汽车
    cv2.rectangle(img, (300, 250), (420, 300), (60, 60, 150), -1)  # 车身
    cv2.circle(img, (330, 300), 20, (30, 30, 30), -1)  # 轮子
    cv2.circle(img, (390, 300), 20, (30, 30, 30), -1)  # 轮子

    # 添加一些噪声使其更真实
    noise = np.random.normal(0, 10, img.shape).astype(np.float64)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img

# 创建图像
color_img = create_sample_image()

# 如果使用真实图像：
# color_img = cv2.imread('your_image.jpg')

# ===================== 步骤1：预处理 =====================

# 1.1 转换为灰度图
gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

# 1.2 降噪（高斯模糊）
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

print("步骤1：预处理完成")
print(f"  灰度图尺寸: {gray.shape}")

# ===================== 步骤2：Sobel边缘检测 =====================

# 2.1 计算X方向梯度
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)

# 2.2 计算Y方向梯度
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# 2.3 计算梯度幅值
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

print("步骤2：Sobel边缘检测完成")

# ===================== 步骤3：后处理 =====================

# 3.1 二值化
_, binary = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

# 3.2 形态学操作（可选，细化边缘）
kernel = np.ones((2, 2), np.uint8)
morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 3.3 边缘叠加到原图
edges_colored = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
edges_colored[morphed > 0] = [0, 255, 0]  # 绿色边缘
overlay = cv2.addWeighted(color_img, 0.7, edges_colored, 0.3, 0)

print("步骤3：后处理完成")

# ===================== 可视化完整流程 =====================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 第一行：处理流程
axes[0, 0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('1. 原始彩色图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('2. 灰度转换', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(blurred, cmap='gray')
axes[0, 2].set_title('3. 高斯模糊降噪', fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(magnitude, cmap='gray')
axes[0, 3].set_title('4. Sobel边缘检测', fontsize=11)
axes[0, 3].axis('off')

# 第二行：后处理和结果
axes[1, 0].imshow(cv2.convertScaleAbs(sobel_x), cmap='gray')
axes[1, 0].set_title('Sobel X（垂直边缘）', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.convertScaleAbs(sobel_y), cmap='gray')
axes[1, 1].set_title('Sobel Y（水平边缘）', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(binary, cmap='gray')
axes[1, 2].set_title('5. 二值化', fontsize=11)
axes[1, 2].axis('off')

axes[1, 3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title('6. 边缘叠加到原图', fontsize=11)
axes[1, 3].axis('off')

plt.suptitle('Sobel边缘检测完整流程', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_complete_pipeline.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n完整流程图已保存为 'sobel_complete_pipeline.png'")
