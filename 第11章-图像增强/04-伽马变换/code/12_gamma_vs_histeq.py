"""
示例12：伽马变换 vs 直方图均衡化
- 对比四种方法：伽马、全局均衡化、CLAHE、伽马+CLAHE
- 统计各方法的均值、标准差、范围
- 直方图对比分析
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def compare_enhancement_methods(image):
    """对比多种增强方法"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 方法1：伽马校正
    gamma = 0.5
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)]).astype(np.uint8)
    gamma_result = cv2.LUT(gray, table)

    # 方法2：直方图均衡化
    hist_eq_result = cv2.equalizeHist(gray)

    # 方法3：CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(gray)

    # 方法4：伽马+CLAHE
    gamma_clahe = clahe.apply(gamma_result)

    return {
        '原图': gray,
        '伽马(γ=0.5)': gamma_result,
        '直方图均衡化': hist_eq_result,
        'CLAHE': clahe_result,
        '伽马+CLAHE': gamma_clahe
    }


# 创建测试图像（暗图像）
test_image = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(test_image, (50, 50), (150, 150), 60, -1)
cv2.rectangle(test_image, (100, 100), (200, 200), 80, -1)
cv2.circle(test_image, (300, 150), 80, 50, -1)
cv2.ellipse(test_image, (200, 250), (100, 40), 30, 0, 360, 70, -1)
noise = np.random.normal(0, 3, test_image.shape).astype(np.int16)
test_image = np.clip(test_image + 20 + noise, 0, 255).astype(np.uint8)

results = compare_enhancement_methods(test_image)

# 可视化
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('增强方法综合对比', fontsize=14, fontweight='bold')

for i, (title, img) in enumerate(results.items()):
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(title, fontsize=10)
    axes[0, i].axis('off')

    axes[1, i].hist(img.flatten(), bins=256, range=[0, 256],
                    alpha=0.7, color='steelblue')
    axes[1, i].set_xlim([0, 256])
    axes[1, i].set_xlabel('灰度值')
    if i == 0:
        axes[1, i].set_ylabel('频率')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '12_gamma_vs_histeq.png'), dpi=150, bbox_inches='tight')
plt.show()

# 统计对比
print("\n图像统计量对比：")
print("-" * 65)
print(f"{'方法':18s} {'均值':>8s} {'标准差':>8s} {'最小值':>8s} {'最大值':>8s}")
print("-" * 65)
for title, img in results.items():
    print(f"{title:18s} {np.mean(img):>8.1f} {np.std(img):>8.1f} "
          f"{np.min(img):>8d} {np.max(img):>8d}")
print("-" * 65)
