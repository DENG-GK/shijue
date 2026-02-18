"""
示例1：展示普通均衡化的问题
- 创建具有不同亮度区域的图像
- 对比全局均衡化和CLAHE的效果
- 说明全局方法在复杂光照下的局限性
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def show_global_equalization_problem():
    """展示全局均衡化的问题"""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 左侧：暗区域
    img[:, :200] = np.random.normal(60, 15, (300, 200)).clip(0, 255)
    cv2.rectangle(img, (30, 50), (170, 250), 100, -1)

    # 右侧：亮区域
    img[:, 200:] = np.random.normal(180, 15, (300, 200)).clip(0, 255)
    cv2.circle(img, (300, 150), 60, 140, -1)

    img = img.astype(np.uint8)

    # 全局均衡化
    global_eq = cv2.equalizeHist(img)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_eq = clahe.apply(img)

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('全局均衡化的问题 vs CLAHE的解决方案', fontsize=14, fontweight='bold')

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('原始图像\n（左暗右亮，不同区域）', fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].hist(img.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    axes[0, 1].set_title('原始直方图', fontsize=11)
    axes[0, 1].set_xlim([0, 255])

    axes[1, 0].imshow(global_eq, cmap='gray')
    axes[1, 0].set_title('全局均衡化\n（某些区域过度增强）', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(clahe_eq, cmap='gray')
    axes[1, 1].set_title('CLAHE\n（局部对比度更好）', fontsize=11)
    axes[1, 1].axis('off')

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '01_global_eq_problem.png'), dpi=150, bbox_inches='tight')
    plt.show()


show_global_equalization_problem()

print("问题分析：")
print("=" * 60)
print("全局均衡化的问题：")
print("1. 某些区域可能被过度增强")
print("2. 某些区域的细节可能丢失")
print("3. 噪声可能被放大")
print("\nCLAHE的解决方案：")
print("1. 将图像分成小块（tiles）")
print("2. 对每个小块分别进行均衡化")
print("3. 使用对比度限制（clip limit）防止过度增强")
print("=" * 60)
