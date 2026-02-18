"""
示例9：2D直方图（H-S直方图）
- 计算HSV空间的2D直方图（色调-饱和度）
- 线性尺度和对数尺度两种显示方式
- 通过2D直方图分析图像的颜色分布
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def compute_2d_histogram(image):
    """计算H-S 2D直方图"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 计算2D直方图
    # H: 0-180, S: 0-256
    hist_2d = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    return hist_2d, hsv


def create_colorful_image():
    """创建多颜色测试图像"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # 不同颜色的区域
    colors = [
        ([0, 0, 255], (50, 50)),      # 红色
        ([0, 255, 0], (150, 50)),     # 绿色
        ([255, 0, 0], (250, 50)),     # 蓝色
        ([0, 255, 255], (100, 150)),  # 黄色
        ([255, 0, 255], (200, 150)),  # 洋红
        ([255, 255, 0], (300, 150)),  # 青色
    ]

    for color, center in colors:
        cv2.circle(img, center, 40, color, -1)

    return img


color_img = create_colorful_image()
hist_2d, hsv = compute_2d_histogram(color_img)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('2D直方图（H-S色调-饱和度）', fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('彩色图像', fontsize=12)
axes[0].axis('off')

# 2D直方图（线性尺度）
im = axes[1].imshow(hist_2d.T, origin='lower', aspect='auto',
                    extent=[0, 180, 0, 256], cmap='jet')
axes[1].set_title('2D直方图（线性尺度）', fontsize=12)
axes[1].set_xlabel('色调 H (0-180)')
axes[1].set_ylabel('饱和度 S (0-256)')
plt.colorbar(im, ax=axes[1])

# 对数尺度显示（更清晰）
hist_log = np.log1p(hist_2d)
im2 = axes[2].imshow(hist_log.T, origin='lower', aspect='auto',
                     extent=[0, 180, 0, 256], cmap='jet')
axes[2].set_title('2D直方图（对数尺度）', fontsize=12)
axes[2].set_xlabel('色调 H (0-180)')
axes[2].set_ylabel('饱和度 S (0-256)')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()

# 保存图片
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_2d_histogram.png'), dpi=150, bbox_inches='tight')
plt.show()

print("2D直方图说明：")
print("- 横轴(H): 色调，表示颜色类型")
print("- 纵轴(S): 饱和度，表示颜色纯度")
print("- 颜色深浅: 该颜色组合出现的频率")
