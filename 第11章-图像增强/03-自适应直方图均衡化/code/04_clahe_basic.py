"""
示例4：CLAHE的基本使用
- cv2.createCLAHE() 创建CLAHE对象
- clahe.apply() 应用到灰度图像
- 对比原图、全局均衡化、CLAHE三种效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def basic_clahe_demo():
    """CLAHE基本演示"""
    # 创建低对比度测试图像
    img = np.zeros((300, 400), dtype=np.uint8)

    for i in range(300):
        img[i, :] = 60 + int(40 * np.sin(i / 30))

    cv2.rectangle(img, (50, 50), (150, 200), 100, -1)
    cv2.circle(img, (280, 150), 60, 80, -1)
    cv2.ellipse(img, (200, 80), (40, 25), 30, 0, 360, 110, -1)

    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(img)

    # 普通均衡化对比
    global_eq = cv2.equalizeHist(img)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CLAHE基本使用', fontsize=14, fontweight='bold')

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(global_eq, cmap='gray')
    axes[0, 1].set_title('全局均衡化', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(result, cmap='gray')
    axes[0, 2].set_title('CLAHE', fontsize=12)
    axes[0, 2].axis('off')

    # 直方图
    hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_global = cv2.calcHist([global_eq], [0], None, [256], [0, 256])
    hist_clahe = cv2.calcHist([result], [0], None, [256], [0, 256])

    axes[1, 0].plot(hist_orig)
    axes[1, 0].fill_between(range(256), hist_orig.flatten(), alpha=0.3)
    axes[1, 0].set_title('原始直方图', fontsize=11)
    axes[1, 0].set_xlim([0, 255])

    axes[1, 1].plot(hist_global, color='orange')
    axes[1, 1].fill_between(range(256), hist_global.flatten(), alpha=0.3, color='orange')
    axes[1, 1].set_title('全局均衡化直方图', fontsize=11)
    axes[1, 1].set_xlim([0, 255])

    axes[1, 2].plot(hist_clahe, color='green')
    axes[1, 2].fill_between(range(256), hist_clahe.flatten(), alpha=0.3, color='green')
    axes[1, 2].set_title('CLAHE直方图', fontsize=11)
    axes[1, 2].set_xlim([0, 255])

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '04_clahe_basic.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("CLAHE使用步骤：")
    print("1. clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))")
    print("2. result = clahe.apply(gray_image)")


basic_clahe_demo()
