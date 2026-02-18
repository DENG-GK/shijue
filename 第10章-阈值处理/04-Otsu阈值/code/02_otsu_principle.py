"""
可视化Otsu算法的工作原理
展示类间方差随阈值变化的曲线
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建双峰图像 =====================

def create_bimodal_image():
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:, :200] = np.random.normal(60, 15, (300, 200)).clip(0, 255)
    img[:, 200:] = np.random.normal(190, 15, (300, 200)).clip(0, 255)
    return img.astype(np.uint8)

# ===================== 可视化Otsu原理 =====================

def visualize_otsu_principle(image):
    """可视化Otsu算法原理"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 计算归一化直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()

    # 计算每个阈值的类间方差
    between_class_variance = np.zeros(256)

    for T in range(256):
        w0 = hist_norm[:T+1].sum()
        w1 = hist_norm[T+1:].sum()

        if w0 == 0 or w1 == 0:
            continue

        indices = np.arange(256)
        mu0 = (indices[:T+1] * hist_norm[:T+1]).sum() / w0
        mu1 = (indices[T+1:] * hist_norm[T+1:]).sum() / w1

        between_class_variance[T] = w0 * w1 * (mu0 - mu1) ** 2

    # 找到最大类间方差对应的阈值
    optimal_T = np.argmax(between_class_variance)

    # OpenCV验证
    cv_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].bar(range(256), hist_norm, width=1, alpha=0.7)
    axes[0, 1].axvline(x=optimal_T, color='r', linestyle='--', linewidth=2,
                       label=f'最佳阈值 T={optimal_T}')
    axes[0, 1].set_title('归一化直方图', fontsize=12)
    axes[0, 1].set_xlabel('像素值')
    axes[0, 1].set_ylabel('概率')
    axes[0, 1].legend()

    axes[1, 0].plot(between_class_variance, 'b-', linewidth=1.5)
    axes[1, 0].axvline(x=optimal_T, color='r', linestyle='--', linewidth=2)
    axes[1, 0].scatter([optimal_T], [between_class_variance[optimal_T]],
                       color='r', s=100, zorder=5, label=f'最大值 T={optimal_T}')
    axes[1, 0].set_title('类间方差 σ²_B(T)', fontsize=12)
    axes[1, 0].set_xlabel('阈值 T')
    axes[1, 0].set_ylabel('方差')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    _, binary = cv2.threshold(gray, optimal_T, 255, cv2.THRESH_BINARY)
    axes[1, 1].imshow(binary, cmap='gray')
    axes[1, 1].set_title(f'Otsu结果 (T={optimal_T})', fontsize=12)
    axes[1, 1].axis('off')

    plt.suptitle('Otsu算法工作原理', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('otsu_principle.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"手动计算的Otsu阈值: {optimal_T}")
    print(f"OpenCV计算的阈值: {cv_thresh:.0f}")
    print(f"最大类间方差: {between_class_variance[optimal_T]:.4f}")

    return optimal_T

# ===================== 运行 =====================

test_img = create_bimodal_image()
optimal_threshold = visualize_otsu_principle(test_img)
