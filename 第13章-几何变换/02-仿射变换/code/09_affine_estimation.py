"""
示例9：仿射变换估计（多点最小二乘）
- estimateAffine2D (RANSAC鲁棒估计)
- estimateAffinePartial2D (仅旋转+缩放+平移)
- 与3点精确解对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建带特征点的测试图像
image = np.zeros((300, 400, 3), dtype=np.uint8)
image[:, :] = [200, 200, 200]

src_pts = np.array([
    [50, 50], [350, 50], [350, 250], [50, 250],
    [200, 100], [200, 200], [100, 150], [300, 150]
], dtype=np.float32)

for i, pt in enumerate(src_pts):
    cv2.circle(image, tuple(pt.astype(int)), 8, (0, 255, 0), -1)
    cv2.putText(image, str(i), (int(pt[0]) + 10, int(pt[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

h, w = image.shape[:2]

# 真实变换
true_M = np.float32([[0.85, -0.2, 50], [0.15, 0.9, 30]])

# 变换点
ones = np.ones((len(src_pts), 1))
src_h = np.hstack([src_pts, ones])
dst_pts = (src_h @ true_M.T).astype(np.float32)

# 添加噪声
np.random.seed(42)
noise = np.random.normal(0, 2, dst_pts.shape).astype(np.float32)
dst_pts_noisy = dst_pts + noise

# 方法1: 3点精确解
M_3pts = cv2.getAffineTransform(src_pts[:3], dst_pts_noisy[:3])

# 方法2: RANSAC鲁棒估计
M_robust, inliers = cv2.estimateAffine2D(src_pts, dst_pts_noisy, method=cv2.RANSAC)

# 方法3: 部分仿射（仅旋转+缩放+平移）
M_partial, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts_noisy)

# 应用变换
transformed_3pts = cv2.warpAffine(image, M_3pts, (w, h))
transformed_robust = cv2.warpAffine(image, M_robust, (w, h))
transformed_partial = cv2.warpAffine(image, M_partial, (w, h))
transformed_true = cv2.warpAffine(image, true_M, (w, h))

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('仿射变换估计方法对比', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('源图像 (带特征点)')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(transformed_true, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('真实变换')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(transformed_3pts, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('3点精确解')
axes[0, 2].axis('off')

axes[0, 3].imshow(cv2.cvtColor(transformed_robust, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title('RANSAC (全部点)')
axes[0, 3].axis('off')

axes[1, 0].imshow(cv2.cvtColor(transformed_partial, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('部分仿射\n(旋转+缩放+平移)')
axes[1, 0].axis('off')

# 点对应关系
axes[1, 1].set_title('点对应关系')
for i in range(len(src_pts)):
    axes[1, 1].plot([src_pts[i, 0], dst_pts_noisy[i, 0]],
                    [src_pts[i, 1], dst_pts_noisy[i, 1]], 'b-', alpha=0.5)
    axes[1, 1].plot(src_pts[i, 0], src_pts[i, 1], 'go', markersize=8)
    axes[1, 1].plot(dst_pts_noisy[i, 0], dst_pts_noisy[i, 1], 'rx', markersize=8)
axes[1, 1].legend(['对应', '源点', '目标点'], loc='upper right', fontsize=8)
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True, alpha=0.3)

# 误差对比
errors = {
    '真实': 0,
    '3点': np.linalg.norm(true_M - M_3pts),
    'RANSAC': np.linalg.norm(true_M - M_robust),
    '部分仿射': np.linalg.norm(true_M - M_partial),
}
axes[1, 2].bar(errors.keys(), errors.values(),
               color=['green', 'red', 'blue', 'orange'])
axes[1, 2].set_ylabel('矩阵Frobenius范数误差')
axes[1, 2].set_title('估计误差对比')

axes[1, 3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_affine_estimation.png'), dpi=150, bbox_inches='tight')
plt.show()

print("变换矩阵对比:")
print(f"真实:     {true_M.flatten()}")
print(f"3点:      {M_3pts.flatten()}")
print(f"RANSAC:   {M_robust.flatten()}")
print(f"部分仿射: {M_partial.flatten()}")
