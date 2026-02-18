"""
高斯模糊预处理提升Otsu效果
对比不同程度的高斯模糊对Otsu阈值的影响
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建带噪声的测试图像 =====================

def create_noisy_image():
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:, :200] = 70
    img[:, 200:] = 180

    cv2.circle(img, (100, 150), 50, 180, -1)
    cv2.rectangle(img, (250, 100), (350, 200), 70, -1)

    # 添加较强的噪声
    noise = np.random.normal(0, 25, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# ===================== 不同模糊程度对比 =====================

def otsu_with_preprocessing(image, blur_sizes=[0, 3, 5, 7]):
    """对比不同程度高斯模糊对Otsu的影响"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    results = []

    for ksize in blur_sizes:
        if ksize == 0:
            processed = gray.copy()
            name = "无模糊"
        else:
            processed = cv2.GaussianBlur(gray, (ksize, ksize), 0)
            name = f"模糊 {ksize}x{ksize}"

        thresh, binary = cv2.threshold(processed, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append({
            'name': name,
            'processed': processed,
            'binary': binary,
            'threshold': thresh
        })

    return results

# ===================== 运行 =====================

noisy_img = create_noisy_image()
results = otsu_with_preprocessing(noisy_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, res in enumerate(results):
    axes[0, i].imshow(res['processed'], cmap='gray')
    axes[0, i].set_title(res['name'], fontsize=11)
    axes[0, i].axis('off')

    axes[1, i].imshow(res['binary'], cmap='gray')
    axes[1, i].set_title(f"T={res['threshold']:.0f}", fontsize=11)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('预处理后', fontsize=12)
axes[1, 0].set_ylabel('Otsu结果', fontsize=12)

plt.suptitle('高斯模糊预处理对Otsu的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gaussian_preprocessing.png', dpi=150, bbox_inches='tight')
plt.show()

print("高斯模糊预处理的效果：")
for res in results:
    print(f"  {res['name']:12s}: Otsu阈值 = {res['threshold']:.0f}")
print("\n推荐：对噪声图像先进行3x3或5x5的高斯模糊")
