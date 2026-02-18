"""
示例1：基本仿射变换
- getAffineTransform 三点定义
- warpAffine 应用变换
- 平移/旋转/缩放/错切效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
image = np.zeros((300, 400, 3), dtype=np.uint8)
image[:, :] = [220, 220, 220]
cv2.rectangle(image, (50, 50), (350, 250), (0, 128, 255), -1)
cv2.putText(image, 'AFFINE', (100, 165), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
cv2.circle(image, (80, 80), 20, (255, 0, 0), -1)

h, w = image.shape[:2]

# 源三角形顶点
src_pts = np.float32([[50, 50], [350, 50], [50, 250]])

# 不同目标三角形 → 不同变换效果
transformations = {
    '原始': np.float32([[50, 50], [350, 50], [50, 250]]),
    '平移': np.float32([[100, 80], [400, 80], [100, 280]]),
    '旋转': np.float32([[100, 150], [300, 50], [0, 250]]),
    '缩放': np.float32([[25, 25], [375, 25], [25, 275]]),
    '错切X': np.float32([[100, 50], [400, 50], [50, 250]]),
    '错切Y': np.float32([[50, 100], [350, 50], [50, 300]]),
}

results = {}
for name, dst_pts in transformations.items():
    if name == '原始':
        results[name] = image.copy()
    else:
        M = cv2.getAffineTransform(src_pts, dst_pts)
        results[name] = cv2.warpAffine(image, M, (w, h))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('基本仿射变换效果', fontsize=14, fontweight='bold')

for i, (name, result) in enumerate(results.items()):
    row, col = i // 3, i % 3
    axes[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(name)
    axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_basic_affine.png'), dpi=150, bbox_inches='tight')
plt.show()

print("基本仿射变换完成！")
print("核心函数: cv2.getAffineTransform(src_pts, dst_pts)")
print("         cv2.warpAffine(image, M, (w, h))")
