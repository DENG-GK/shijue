"""
彩色图像的阈值处理
对比三种不同的彩色图像阈值处理方法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 彩色图像阈值处理函数 =====================

def threshold_color_image(image, thresh, method='gray'):
    """
    彩色图像阈值处理

    Parameters:
    -----------
    image : numpy.ndarray - 输入的彩色图像 (BGR格式)
    thresh : int - 阈值
    method : str - 'gray'转灰度处理, 'channels'逐通道处理, 'hsv'在HSV空间处理
    """
    if method == 'gray':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        return result

    elif method == 'channels':
        channels = cv2.split(image)
        binary_channels = []
        for ch in channels:
            _, binary = cv2.threshold(ch, thresh, 255, cv2.THRESH_BINARY)
            binary_channels.append(binary)
        result = cv2.merge(binary_channels)
        return result

    elif method == 'hsv':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        _, v_binary = cv2.threshold(v, thresh, 255, cv2.THRESH_BINARY)
        return v_binary

    else:
        raise ValueError(f"未知方法: {method}")

# ===================== 创建彩色测试图像 =====================

def create_color_test_image():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:, :] = [50, 50, 50]  # 深灰色背景

    # 添加彩色形状
    cv2.circle(img, (100, 150), 60, (255, 0, 0), -1)    # 蓝色圆
    cv2.rectangle(img, (180, 80), (280, 220), (0, 255, 0), -1)  # 绿色矩形
    cv2.ellipse(img, (350, 150), (40, 70), 0, 0, 360, (0, 0, 255), -1)  # 红色椭圆

    return img

color_img = create_color_test_image()
T = 100

# ===================== 三种处理方法 =====================

result_gray = threshold_color_image(color_img, T, 'gray')
result_channels = threshold_color_image(color_img, T, 'channels')
result_hsv = threshold_color_image(color_img, T, 'hsv')

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始彩色图像', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(result_gray, cmap='gray')
axes[0, 1].set_title('方法1: 灰度法\n(先转灰度再处理)', fontsize=11)
axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(result_channels, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('方法2: 逐通道法\n(分别处理每个通道)', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(result_hsv, cmap='gray')
axes[1, 1].set_title('方法3: HSV法\n(处理V通道)', fontsize=11)
axes[1, 1].axis('off')

plt.suptitle('彩色图像阈值处理方法对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('color_threshold.png', dpi=150, bbox_inches='tight')
plt.show()

print("彩色图像阈值处理方法对比：")
print("- Gray: 简单快速，但丢失颜色信息")
print("- Channels: 保留颜色信息，但可能产生意外的颜色组合")
print("- HSV: 只处理亮度通道，通常效果较好")
