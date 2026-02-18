"""
示例2：手动实现直方图均衡化
- 逐步实现：计算直方图 → CDF → 归一化 → 映射
- 与OpenCV的cv2.equalizeHist()结果对比
- 可视化中间过程（直方图、CDF、差异图）
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def equalize_histogram_manual(image):
    """
    手动实现直方图均衡化

    步骤：
    1. 计算直方图
    2. 计算累积分布函数（CDF）
    3. 归一化CDF
    4. 映射像素值
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 步骤1：计算直方图
    hist = np.zeros(256)
    for pixel in gray.flatten():
        hist[pixel] += 1

    # 步骤2：计算CDF
    cdf = np.cumsum(hist)

    # 步骤3：归一化CDF（映射到0-255）
    cdf_min = cdf[cdf > 0].min()  # 第一个非零值
    cdf_normalized = (cdf - cdf_min) / (gray.size - cdf_min) * 255

    # 处理可能的NaN
    cdf_normalized = np.nan_to_num(cdf_normalized, nan=0)

    # 步骤4：映射像素值
    equalized = cdf_normalized[gray].astype(np.uint8)

    return equalized, hist, cdf, cdf_normalized


# 创建测试图像
test_img = np.random.normal(80, 25, (300, 400)).clip(0, 255).astype(np.uint8)

# 手动实现
manual_result, hist, cdf, cdf_norm = equalize_histogram_manual(test_img)

# OpenCV实现
opencv_result = cv2.equalizeHist(test_img)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('手动实现 vs OpenCV 直方图均衡化', fontsize=14, fontweight='bold')

axes[0, 0].imshow(test_img, cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].bar(range(256), hist, width=1, color='blue', alpha=0.7)
axes[0, 1].set_title('直方图', fontsize=11)
axes[0, 1].set_xlim([0, 255])
axes[0, 1].set_xlabel('像素值')

axes[0, 2].plot(cdf / cdf.max() * 255, color='red', linewidth=2, label='CDF')
axes[0, 2].plot(cdf_norm, color='green', linewidth=2, linestyle='--', label='归一化CDF')
axes[0, 2].set_title('累积分布函数 (CDF)', fontsize=11)
axes[0, 2].set_xlim([0, 255])
axes[0, 2].legend()

axes[1, 0].imshow(manual_result, cmap='gray')
axes[1, 0].set_title('手动实现结果', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(opencv_result, cmap='gray')
axes[1, 1].set_title('OpenCV实现结果', fontsize=11)
axes[1, 1].axis('off')

diff = np.abs(manual_result.astype(int) - opencv_result.astype(int))
axes[1, 2].imshow(diff, cmap='hot')
axes[1, 2].set_title(f'差异图 (最大差异: {diff.max()})', fontsize=11)
axes[1, 2].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_manual_equalization.png'), dpi=150, bbox_inches='tight')
plt.show()

print("验证结果：")
print(f"最大差异: {diff.max()} 像素级别")
print(f"平均差异: {diff.mean():.4f}")
print(f"结果一致: {np.allclose(manual_result, opencv_result, atol=1)}")
