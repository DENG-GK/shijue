"""
示例11：CLAHE与普通均衡化的综合对比
- 局部对比度不均、暗图像、带噪声图像三种场景
- 对比原图、全局均衡化、CLAHE效果
- 总结两种方法的适用场景
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def comprehensive_comparison():
    """综合对比CLAHE和普通均衡化"""
    test_cases = {}

    # 1. 局部对比度不均
    img1 = np.zeros((200, 300), dtype=np.uint8)
    img1[:, :150] = np.random.normal(50, 10, (200, 150)).clip(0, 255)
    img1[:, 150:] = np.random.normal(180, 10, (200, 150)).clip(0, 255)
    test_cases['局部对比度不均'] = img1.astype(np.uint8)

    # 2. 暗图像
    img2 = np.random.normal(40, 20, (200, 300)).clip(0, 255).astype(np.uint8)
    cv2.rectangle(img2, (50, 30), (150, 170), 80, -1)
    test_cases['暗图像'] = img2

    # 3. 带噪声
    img3 = np.random.normal(100, 30, (200, 300)).clip(0, 255).astype(np.uint8)
    noise = np.random.normal(0, 20, img3.shape)
    img3 = np.clip(img3 + noise, 0, 255).astype(np.uint8)
    test_cases['带噪声图像'] = img3

    fig, axes = plt.subplots(len(test_cases), 3, figsize=(12, 4 * len(test_cases)))
    fig.suptitle('CLAHE vs 全局均衡化 综合对比', fontsize=14, fontweight='bold')

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for i, (name, img) in enumerate(test_cases.items()):
        global_eq = cv2.equalizeHist(img)
        clahe_eq = clahe.apply(img)

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'原始: {name}', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(global_eq, cmap='gray')
        axes[i, 1].set_title('全局均衡化', fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(clahe_eq, cmap='gray')
        axes[i, 2].set_title('CLAHE', fontsize=10)
        axes[i, 2].axis('off')

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '11_comprehensive_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()


comprehensive_comparison()

print("\nCLAHE vs 普通均衡化 总结：")
print("=" * 60)
print(f"{'方面':15s} {'普通均衡化':18s} {'CLAHE':18s}")
print("-" * 60)
print(f"{'处理范围':15s} {'全局':18s} {'局部（分块）':18s}")
print(f"{'对比度控制':15s} {'无':18s} {'clipLimit可控':18s}")
print(f"{'噪声放大':15s} {'严重':18s} {'可控':18s}")
print(f"{'局部细节':15s} {'可能丢失':18s} {'保留较好':18s}")
print(f"{'计算速度':15s} {'快':18s} {'稍慢':18s}")
print(f"{'适用场景':15s} {'整体偏暗/亮':18s} {'复杂光照条件':18s}")
print("=" * 60)
