"""
示例3：可视化像素映射函数
- 展示均衡化的像素值映射关系
- 对比恒等映射（无变化）与均衡化映射
- 分析暗图像映射函数的特点
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_mapping(image):
    """可视化直方图均衡化的映射函数"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 计算直方图和CDF
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    cdf = np.cumsum(hist)
    cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255

    # 创建映射表
    mapping = cdf_normalized.astype(np.uint8)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('像素值映射函数可视化', fontsize=14, fontweight='bold')

    # 映射函数
    axes[0].plot(range(256), mapping, 'b-', linewidth=2, label='均衡化映射')
    axes[0].plot([0, 255], [0, 255], 'r--', linewidth=1, label='恒等映射（无变化）')
    axes[0].set_title('像素值映射函数', fontsize=12)
    axes[0].set_xlabel('输入像素值')
    axes[0].set_ylabel('输出像素值')
    axes[0].set_xlim([0, 255])
    axes[0].set_ylim([0, 255])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')

    # 直方图对比
    equalized = cv2.equalizeHist(gray)
    hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256]).flatten()

    axes[1].plot(hist, 'b-', alpha=0.7, label='原始')
    axes[1].plot(hist_eq, 'g-', alpha=0.7, label='均衡化后')
    axes[1].set_title('直方图对比', fontsize=12)
    axes[1].set_xlabel('像素值')
    axes[1].set_ylabel('频率')
    axes[1].set_xlim([0, 255])
    axes[1].legend()

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '03_mapping_function.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return mapping


# 创建测试图像（暗图像）
dark_img = np.random.normal(60, 30, (300, 400)).clip(0, 255).astype(np.uint8)

mapping = visualize_mapping(dark_img)

print("映射函数分析：")
print(f"输入0映射到:   {mapping[0]}")
print(f"输入128映射到: {mapping[128]}")
print(f"输入255映射到: {mapping[255]}")
print("\n暗图像特点：映射曲线在低值区域上升快，")
print("表示原本集中的暗像素被拉伸到更宽的范围")
