"""
示例3：五种阈值类型的对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

gradient = np.tile(np.arange(256, dtype=np.uint8), (100, 1))

threshold_types = [
    (cv2.THRESH_BINARY, 'BINARY', 'dst = (src > T) ? maxVal : 0'),
    (cv2.THRESH_BINARY_INV, 'BINARY_INV', 'dst = (src > T) ? 0 : maxVal'),
    (cv2.THRESH_TRUNC, 'TRUNC', 'dst = (src > T) ? T : src'),
    (cv2.THRESH_TOZERO, 'TOZERO', 'dst = (src > T) ? src : 0'),
    (cv2.THRESH_TOZERO_INV, 'TOZERO_INV', 'dst = (src > T) ? 0 : src'),
]

T = 127

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

axes[0].imshow(gradient, cmap='gray')
axes[0].set_title('Original Gradient\n(0 -> 255)', fontsize=12)
axes[0].axvline(x=T, color='r', linestyle='--', linewidth=2)
axes[0].text(T+5, 50, f'T={T}', color='r', fontsize=10)
axes[0].axis('off')

for i, (thresh_type, name, formula) in enumerate(threshold_types, 1):
    _, result = cv2.threshold(gradient, T, 255, thresh_type)
    axes[i].imshow(result, cmap='gray')
    axes[i].set_title(f'{name}\n{formula}', fontsize=10)
    axes[i].axvline(x=T, color='r', linestyle='--', linewidth=2)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('threshold_types_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("阈值类型对比图已保存为 threshold_types_comparison.png")
