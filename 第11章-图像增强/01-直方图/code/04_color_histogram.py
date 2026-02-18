"""
示例4：彩色图像的三通道直方图
- 创建彩色测试图像
- 分别计算B、G、R三个通道的直方图
- 叠加显示三通道直方图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def color_histogram(image):
    """计算彩色图像的三通道直方图"""
    # OpenCV读取的是BGR格式
    colors = ('b', 'g', 'r')
    channel_names = ('Blue 蓝色', 'Green 绿色', 'Red 红色')

    histograms = []
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histograms.append((name, color, hist))

    return histograms


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


color_img = create_color_image()
histograms = color_histogram(color_img)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('彩色图像的三通道直方图', fontsize=14, fontweight='bold')

# 显示图像（转换为RGB用于matplotlib显示）
axes[0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('彩色图像 (BGR→RGB)', fontsize=12)
axes[0].axis('off')

# 显示三通道直方图
for name, color, hist in histograms:
    axes[1].plot(hist, color=color, label=name, linewidth=1.5)

axes[1].set_title('BGR三通道直方图', fontsize=12)
axes[1].set_xlabel('像素值')
axes[1].set_ylabel('频率')
axes[1].set_xlim([0, 255])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_color_histogram.png'), dpi=150, bbox_inches='tight')
plt.show()

print("彩色直方图说明：")
print("- 蓝色曲线(B): 蓝色通道的像素分布")
print("- 绿色曲线(G): 绿色通道的像素分布")
print("- 红色曲线(R): 红色通道的像素分布")
print("- 各通道直方图的峰值位置反映了该通道的主要强度")
