"""
示例5：HSV颜色空间的直方图
- 将BGR图像转换为HSV空间
- 分别计算H（色调）、S（饱和度）、V（明度）通道直方图
- 可视化各通道图像和直方图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_color_image():
    """创建彩色测试图像"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # 红色区域
    img[50:150, 50:150] = [0, 0, 200]
    # 绿色区域
    img[50:150, 150:250] = [0, 200, 0]
    # 蓝色区域
    img[50:150, 250:350] = [200, 0, 0]
    # 混合区域
    img[150:250, 100:200] = [100, 150, 200]  # 偏橙色
    img[150:250, 200:300] = [200, 100, 50]   # 偏青色

    # 添加噪声
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def hsv_histogram(image):
    """计算HSV空间的直方图"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # H通道：色调（0-179）
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])

    # S通道：饱和度（0-255）
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])

    # V通道：明度（0-255）
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])

    return hist_h, hist_s, hist_v, hsv


# 创建彩色图像
color_img = create_color_image()
hist_h, hist_s, hist_v, hsv = hsv_histogram(color_img)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('HSV颜色空间直方图', fontsize=16, fontweight='bold')

# 原图
axes[0, 0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原图 (BGR→RGB)', fontsize=11)
axes[0, 0].axis('off')

# HSV各通道
h, s, v = cv2.split(hsv)

axes[0, 1].imshow(h, cmap='hsv')
axes[0, 1].set_title('H通道 (色调)', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(s, cmap='gray')
axes[0, 2].set_title('S通道 (饱和度)', fontsize=11)
axes[0, 2].axis('off')

# 直方图
axes[1, 0].plot(hist_h, color='purple', linewidth=1.5)
axes[1, 0].fill_between(range(180), hist_h.flatten(), alpha=0.3, color='purple')
axes[1, 0].set_title('色调直方图 (0-179)', fontsize=11)
axes[1, 0].set_xlim([0, 179])
axes[1, 0].set_xlabel('H值')
axes[1, 0].set_ylabel('频率')

axes[1, 1].plot(hist_s, color='orange', linewidth=1.5)
axes[1, 1].fill_between(range(256), hist_s.flatten(), alpha=0.3, color='orange')
axes[1, 1].set_title('饱和度直方图', fontsize=11)
axes[1, 1].set_xlim([0, 255])
axes[1, 1].set_xlabel('S值')

axes[1, 2].plot(hist_v, color='green', linewidth=1.5)
axes[1, 2].fill_between(range(256), hist_v.flatten(), alpha=0.3, color='green')
axes[1, 2].set_title('明度直方图', fontsize=11)
axes[1, 2].set_xlim([0, 255])
axes[1, 2].set_xlabel('V值')

plt.tight_layout()

# 保存图片
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_hsv_histogram.png'), dpi=150, bbox_inches='tight')
plt.show()

print("HSV直方图说明：")
print("- H (Hue): 表示颜色类型，范围0-179")
print("- S (Saturation): 表示颜色纯度，范围0-255")
print("- V (Value): 表示亮度，范围0-255")
