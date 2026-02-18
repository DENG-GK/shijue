"""
示例6：使用掩码计算感兴趣区域的直方图
- 创建圆形掩码限定ROI区域
- 对比全图直方图与掩码区域直方图
- 分析掩码区域的像素分布特征
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def roi_histogram(image, mask):
    """计算掩码区域的直方图"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 全图直方图
    hist_full = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # 掩码区域直方图
    hist_mask = cv2.calcHist([gray], [0], mask, [256], [0, 256])

    return hist_full, hist_mask


# 创建测试图像
img = np.zeros((300, 400), dtype=np.uint8)
img[:] = 100  # 背景灰度
cv2.circle(img, (200, 150), 80, 200, -1)  # 亮色圆形
noise = np.random.normal(0, 10, img.shape)
img = np.clip(img + noise, 0, 255).astype(np.uint8)

# 创建圆形掩码
mask = np.zeros(img.shape, dtype=np.uint8)
cv2.circle(mask, (200, 150), 80, 255, -1)

# 计算直方图
hist_full, hist_mask = roi_histogram(img, mask)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('掩码ROI区域直方图', fontsize=16, fontweight='bold')

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(mask, cmap='gray')
axes[0, 1].set_title('掩码 (ROI)', fontsize=12)
axes[0, 1].axis('off')

axes[1, 0].plot(hist_full, color='blue', label='全图')
axes[1, 0].fill_between(range(256), hist_full.flatten(), alpha=0.3)
axes[1, 0].set_title('全图直方图', fontsize=12)
axes[1, 0].set_xlim([0, 255])
axes[1, 0].set_xlabel('像素值')
axes[1, 0].set_ylabel('频率')
axes[1, 0].legend()

axes[1, 1].plot(hist_mask, color='red', label='掩码区域')
axes[1, 1].fill_between(range(256), hist_mask.flatten(), alpha=0.3, color='red')
axes[1, 1].set_title('掩码区域直方图', fontsize=12)
axes[1, 1].set_xlim([0, 255])
axes[1, 1].set_xlabel('像素值')
axes[1, 1].set_ylabel('频率')
axes[1, 1].legend()

plt.tight_layout()

# 保存图片
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_roi_histogram.png'), dpi=150, bbox_inches='tight')
plt.show()

print("掩码直方图分析：")
print(f"全图像素数:   {hist_full.sum():.0f}")
print(f"掩码区域像素: {hist_mask.sum():.0f}")
print(f"掩码区域占比: {hist_mask.sum() / hist_full.sum() * 100:.1f}%")
