"""
示例5：彩色图像均衡化的错误与正确方法
- 错误：分别对BGR三通道均衡化（导致颜色失真）
- 正确1：在YCrCb空间仅对Y通道均衡化
- 正确2：在HSV空间仅对V通道均衡化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_color_image():
    """创建测试彩色图像"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # 创建渐变背景
    for i in range(300):
        for j in range(400):
            img[i, j] = [int(50 + 100 * i / 300),
                        int(80 + 80 * j / 400),
                        int(60 + 60 * (i + j) / 700)]

    # 添加彩色形状
    cv2.circle(img, (100, 150), 50, (200, 100, 50), -1)
    cv2.rectangle(img, (200, 80), (320, 220), (50, 180, 120), -1)

    return img


def equalize_color_wrong(image):
    """错误方法：直接对每个通道均衡化"""
    b, g, r = cv2.split(image)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    return cv2.merge([b_eq, g_eq, r_eq])


def equalize_color_correct(image):
    """正确方法：在YCrCb/HSV空间的亮度通道均衡化"""
    # 方法1：YCrCb空间
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    result_ycrcb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # 方法2：HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    result_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result_ycrcb, result_hsv


# 创建测试图像
color_img = create_color_image()

# 不同方法
wrong_result = equalize_color_wrong(color_img)
correct_ycrcb, correct_hsv = equalize_color_correct(color_img)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('彩色图像直方图均衡化：错误方法 vs 正确方法', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(wrong_result, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('错误: 分别对BGR通道均衡化\n（颜色严重失真!）', fontsize=11)
axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(correct_ycrcb, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('正确: YCrCb空间（仅Y通道）\n（推荐方法）', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(correct_hsv, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('正确: HSV空间（仅V通道）', fontsize=11)
axes[1, 1].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_color_equalization.png'), dpi=150, bbox_inches='tight')
plt.show()

print("彩色图像直方图均衡化：")
print("=" * 60)
print("× 错误方法: 分别对B、G、R通道均衡化 → 会导致颜色失真")
print("√ 正确方法1: 转换到YCrCb空间，只对Y通道均衡化")
print("√ 正确方法2: 转换到HSV空间，只对V通道均衡化")
print("=" * 60)
