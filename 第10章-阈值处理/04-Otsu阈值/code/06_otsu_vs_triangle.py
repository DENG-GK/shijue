"""
Otsu vs Triangle 阈值对比
分别在双峰和单峰图像上测试两种方法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建测试图像 =====================

def create_bimodal_image():
    """双峰分布图像"""
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:, :200] = np.random.normal(60, 15, (300, 200)).clip(0, 255)
    img[:, 200:] = np.random.normal(190, 15, (300, 200)).clip(0, 255)
    return img.astype(np.uint8)

def create_unimodal_image():
    """单峰分布图像"""
    img = np.random.normal(180, 30, (300, 400)).clip(0, 255).astype(np.uint8)

    # 添加少量暗色目标
    cv2.circle(img, (100, 150), 40, 30, -1)
    cv2.rectangle(img, (250, 100), (350, 200), 50, -1)

    return img

# ===================== 对比函数 =====================

def compare_otsu_triangle(image):
    """比较Otsu和Triangle阈值"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    otsu_thresh, otsu_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    triangle_thresh, triangle_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    return {
        'original': gray,
        'otsu': (otsu_thresh, otsu_binary),
        'triangle': (triangle_thresh, triangle_binary)
    }

# ===================== 测试 =====================

bimodal_img = create_bimodal_image()
unimodal_img = create_unimodal_image()

bimodal_results = compare_otsu_triangle(bimodal_img)
unimodal_results = compare_otsu_triangle(unimodal_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 双峰图像
axes[0, 0].imshow(bimodal_results['original'], cmap='gray')
axes[0, 0].set_title('双峰图像', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].hist(bimodal_results['original'].ravel(), 256, [0, 256])
axes[0, 1].set_title('直方图 (两个峰)', fontsize=11)
axes[0, 1].axvline(x=bimodal_results['otsu'][0], color='r', linestyle='--',
                   label=f"Otsu={bimodal_results['otsu'][0]:.0f}")
axes[0, 1].axvline(x=bimodal_results['triangle'][0], color='g', linestyle=':',
                   label=f"Triangle={bimodal_results['triangle'][0]:.0f}")
axes[0, 1].legend(fontsize=8)

axes[0, 2].imshow(bimodal_results['otsu'][1], cmap='gray')
axes[0, 2].set_title(f"Otsu (T={bimodal_results['otsu'][0]:.0f})", fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(bimodal_results['triangle'][1], cmap='gray')
axes[0, 3].set_title(f"Triangle (T={bimodal_results['triangle'][0]:.0f})", fontsize=11)
axes[0, 3].axis('off')

# 单峰图像
axes[1, 0].imshow(unimodal_results['original'], cmap='gray')
axes[1, 0].set_title('单峰图像', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].hist(unimodal_results['original'].ravel(), 256, [0, 256])
axes[1, 1].set_title('直方图 (一个峰)', fontsize=11)
axes[1, 1].axvline(x=unimodal_results['otsu'][0], color='r', linestyle='--',
                   label=f"Otsu={unimodal_results['otsu'][0]:.0f}")
axes[1, 1].axvline(x=unimodal_results['triangle'][0], color='g', linestyle=':',
                   label=f"Triangle={unimodal_results['triangle'][0]:.0f}")
axes[1, 1].legend(fontsize=8)

axes[1, 2].imshow(unimodal_results['otsu'][1], cmap='gray')
axes[1, 2].set_title(f"Otsu (T={unimodal_results['otsu'][0]:.0f})", fontsize=11)
axes[1, 2].axis('off')

axes[1, 3].imshow(unimodal_results['triangle'][1], cmap='gray')
axes[1, 3].set_title(f"Triangle (T={unimodal_results['triangle'][0]:.0f})", fontsize=11)
axes[1, 3].axis('off')

plt.suptitle('Otsu vs Triangle 阈值对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('otsu_vs_triangle.png', dpi=150, bbox_inches='tight')
plt.show()

print("Otsu vs Triangle 比较：")
print("=" * 50)
print("Otsu:     最适合双峰分布的图像")
print("Triangle: 最适合单峰分布的图像（少量目标）")
print("=" * 50)
