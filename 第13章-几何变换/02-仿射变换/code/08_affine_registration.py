"""
示例8：图像配准预处理
- 已知变换创建变换图像
- getAffineTransform 估计变换
- invertAffineTransform 求逆对齐
- 叠加对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建参考图像
source = np.zeros((300, 400, 3), dtype=np.uint8)
source[:, :] = [200, 200, 200]
cv2.circle(source, (100, 100), 30, (255, 0, 0), -1)
cv2.circle(source, (300, 100), 30, (0, 255, 0), -1)
cv2.circle(source, (200, 250), 30, (0, 0, 255), -1)
cv2.rectangle(source, (150, 80), (250, 180), (255, 255, 0), 3)

h, w = source.shape[:2]

# 已知变换
true_transform = np.float32([[0.9, -0.15, 30], [0.1, 0.95, 20]])
target = cv2.warpAffine(source, true_transform, (w, h))

# 特征点对应
src_pts = np.float32([[100, 100], [300, 100], [200, 250]])
ones = np.ones((3, 1))
src_h = np.hstack([src_pts, ones])
dst_pts = (src_h @ true_transform.T).astype(np.float32)

# 估计变换
estimated_M = cv2.getAffineTransform(src_pts, dst_pts)
M_inv = cv2.invertAffineTransform(estimated_M)
aligned = cv2.warpAffine(target, M_inv, (w, h))

# 误差
error = cv2.absdiff(source, aligned)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('仿射变换配准预处理', fontsize=14, fontweight='bold')

# 标记源点
source_marked = source.copy()
for i, pt in enumerate(src_pts.astype(np.int32)):
    cv2.circle(source_marked, tuple(pt), 8, (0, 255, 255), 2)
    cv2.putText(source_marked, str(i + 1), (pt[0] + 10, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
axes[0, 0].imshow(cv2.cvtColor(source_marked, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('源图像 (参考)')
axes[0, 0].axis('off')

target_marked = target.copy()
for i, pt in enumerate(dst_pts.astype(np.int32)):
    cv2.circle(target_marked, tuple(pt), 8, (0, 255, 255), 2)
axes[0, 1].imshow(cv2.cvtColor(target_marked, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('目标图像 (变换后)')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('对齐结果')
axes[0, 2].axis('off')

# 叠加对比
overlay_before = cv2.addWeighted(source, 0.5, target, 0.5, 0)
axes[1, 0].imshow(cv2.cvtColor(overlay_before, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('对齐前叠加')
axes[1, 0].axis('off')

overlay_after = cv2.addWeighted(source, 0.5, aligned, 0.5, 0)
axes[1, 1].imshow(cv2.cvtColor(overlay_after, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('对齐后叠加')
axes[1, 1].axis('off')

error_enhanced = cv2.convertScaleAbs(error, alpha=5)
axes[1, 2].imshow(cv2.cvtColor(error_enhanced, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f'误差 (5x)\n最大: {error.max()}')
axes[1, 2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_affine_registration.png'), dpi=150, bbox_inches='tight')
plt.show()

print("真实变换矩阵:")
print(true_transform)
print("\n估计变换矩阵:")
print(estimated_M)
print(f"\n矩阵差异: {np.abs(true_transform - estimated_M).max():.6f}")
