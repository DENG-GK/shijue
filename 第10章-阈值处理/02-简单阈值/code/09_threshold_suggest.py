"""
阈值选择辅助工具
根据图像直方图自动建议合适的阈值
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 阈值建议函数 =====================

def suggest_threshold(image):
    """
    根据图像直方图建议阈值
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 方法1：使用均值
    mean_val = np.mean(gray)

    # 方法2：使用中值
    median_val = np.median(gray)

    # 方法3：使用Otsu自动阈值
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 方法4：使用Triangle自动阈值
    triangle_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 显示图像
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('输入图像', fontsize=12)
    axes[0].axis('off')

    # 显示直方图和建议阈值
    axes[1].plot(hist, 'b-', linewidth=1)
    axes[1].axvline(x=mean_val, color='r', linestyle='-', label=f'均值: {mean_val:.0f}')
    axes[1].axvline(x=median_val, color='g', linestyle='--', label=f'中值: {median_val:.0f}')
    axes[1].axvline(x=otsu_thresh, color='m', linestyle='-.', label=f'Otsu: {otsu_thresh:.0f}')
    axes[1].axvline(x=triangle_thresh, color='c', linestyle=':', label=f'Triangle: {triangle_thresh:.0f}')
    axes[1].set_title('直方图与建议阈值', fontsize=12)
    axes[1].set_xlabel('像素值')
    axes[1].set_ylabel('频率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('threshold_suggestion.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n阈值建议：")
    print(f"  均值法:     {mean_val:.0f}")
    print(f"  中值法:     {median_val:.0f}")
    print(f"  Otsu法:     {otsu_thresh:.0f}")
    print(f"  Triangle法: {triangle_thresh:.0f}")

    return {
        'mean': mean_val,
        'median': median_val,
        'otsu': otsu_thresh,
        'triangle': triangle_thresh
    }

# ===================== 创建测试图像 =====================

test_img = np.zeros((300, 400), dtype=np.uint8)
test_img[:, :200] = np.random.normal(70, 20, (300, 200)).clip(0, 255)
test_img[:, 200:] = np.random.normal(180, 20, (300, 200)).clip(0, 255)
test_img = test_img.astype(np.uint8)

# 获取阈值建议
thresholds = suggest_threshold(test_img)

print("\n图像已保存为 'threshold_suggestion.png'")
