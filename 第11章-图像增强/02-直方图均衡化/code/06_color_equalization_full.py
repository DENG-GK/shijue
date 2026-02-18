"""
示例6：完整的彩色图像直方图均衡化
- 支持三种颜色空间：YCrCb、HSV、LAB
- 封装为通用函数，支持method参数切换
- 对比三种方法的效果差异
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def equalize_color_image(image, method='ycrcb'):
    """
    彩色图像直方图均衡化

    Parameters:
    -----------
    image : numpy.ndarray
        输入彩色图像 (BGR格式)
    method : str
        'ycrcb' - 在YCrCb空间均衡化Y通道 (推荐)
        'hsv' - 在HSV空间均衡化V通道
        'lab' - 在LAB空间均衡化L通道
    """
    if method == 'ycrcb':
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    elif method == 'hsv':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif method == 'lab':
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    else:
        raise ValueError(f"未知方法: {method}")

    return result


# 创建低对比度彩色图像
def create_low_contrast_color():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    base = 100
    range_val = 50

    for i in range(300):
        for j in range(400):
            img[i, j] = [base + int(range_val * np.sin(i / 30)),
                        base + int(range_val * np.cos(j / 40)),
                        base + int(range_val * np.sin((i + j) / 50))]

    cv2.circle(img, (150, 150), 60, (140, 100, 80), -1)
    cv2.rectangle(img, (220, 80), (350, 220), (80, 130, 110), -1)

    return img


color_img = create_low_contrast_color()

# 三种方法
result_ycrcb = equalize_color_image(color_img, 'ycrcb')
result_hsv = equalize_color_image(color_img, 'hsv')
result_lab = equalize_color_image(color_img, 'lab')

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('三种颜色空间的直方图均衡化对比', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始图像（低对比度）', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(result_ycrcb, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('YCrCb方法（推荐）', fontsize=12)
axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(result_hsv, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('HSV方法', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(result_lab, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('LAB方法', fontsize=12)
axes[1, 1].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_color_equalization_full.png'), dpi=150, bbox_inches='tight')
plt.show()

print("三种方法对比：")
print("- YCrCb: 最常用，效果稳定")
print("- HSV: 效果类似，但可能在极端情况下产生伪影")
print("- LAB: 感知上更均匀，适合需要精确颜色保持的场景")
