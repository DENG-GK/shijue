"""
示例2：cv2.calcHist() 基本使用
- 计算灰度图像的直方图
- 了解 cv2.calcHist() 的参数和返回值
- 可视化直方图并输出统计信息
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def compute_histogram(image):
    """计算灰度图像的直方图"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 使用OpenCV计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    return hist


# 创建测试图像
img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 250), 100, -1)
cv2.circle(img, (300, 150), 80, 200, -1)
noise = np.random.normal(0, 10, img.shape)
img = np.clip(img + noise, 0, 255).astype(np.uint8)

# 计算直方图
hist = compute_histogram(img)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('cv2.calcHist() 基本使用', fontsize=14, fontweight='bold')

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

axes[1].plot(hist, color='blue', linewidth=1)
axes[1].fill_between(range(256), hist.flatten(), alpha=0.3)
axes[1].set_title('灰度直方图', fontsize=12)
axes[1].set_xlabel('像素值 (0-255)')
axes[1].set_ylabel('频率')
axes[1].set_xlim([0, 255])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_calchist_basic.png'), dpi=150, bbox_inches='tight')
plt.show()

# 打印统计信息
print("cv2.calcHist() 参数说明：")
print("=" * 50)
print("hist = cv2.calcHist(images, channels, mask, histSize, ranges)")
print()
print(f"直方图形状: {hist.shape}")
print(f"像素总数: {hist.sum():.0f}")
print(f"图像尺寸: {img.shape[0]} x {img.shape[1]} = {img.size}")
print(f"最大频率: {hist.max():.0f} (灰度值 {hist.argmax()})")
