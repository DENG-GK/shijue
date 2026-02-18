"""
Otsu与不同阈值类型的组合
THRESH_BINARY + THRESH_OTSU vs THRESH_BINARY_INV + THRESH_OTSU
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== Otsu组合测试 =====================

def otsu_with_different_types(image):
    """Otsu与不同阈值类型组合"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Otsu + BINARY
    thresh1, binary = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu + BINARY_INV
    thresh2, binary_inv = cv2.threshold(gray, 0, 255,
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return {
        'original': gray,
        'binary': (thresh1, binary),
        'binary_inv': (thresh2, binary_inv)
    }

# ===================== 创建测试图像 =====================

test_img = np.zeros((200, 300), dtype=np.uint8)
cv2.circle(test_img, (150, 100), 60, 200, -1)
cv2.rectangle(test_img, (20, 20), (80, 80), 180, -1)
noise = np.random.normal(0, 15, test_img.shape)
test_img = np.clip(test_img + noise, 0, 255).astype(np.uint8)

results = otsu_with_different_types(test_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(results['original'], cmap='gray')
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

axes[1].imshow(results['binary'][1], cmap='gray')
axes[1].set_title(f"BINARY + OTSU\n(T={results['binary'][0]:.0f})", fontsize=11)
axes[1].axis('off')

axes[2].imshow(results['binary_inv'][1], cmap='gray')
axes[2].set_title(f"BINARY_INV + OTSU\n(T={results['binary_inv'][0]:.0f})", fontsize=11)
axes[2].axis('off')

plt.suptitle('Otsu与不同阈值类型的组合', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('otsu_combinations.png', dpi=150, bbox_inches='tight')
plt.show()

print("Otsu阈值组合使用：")
print("- cv2.THRESH_BINARY + cv2.THRESH_OTSU: 标准Otsu")
print("- cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU: 反向Otsu")
print(f"\n两种方式使用相同的阈值: {results['binary'][0]:.0f}")
