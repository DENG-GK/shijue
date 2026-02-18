"""
示例6：多点单应性估计
- findHomography (RANSAC / LMEDS)
- 与4点精确解对比
- 内点/外点可视化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建特征点图像
image = np.zeros((300, 400, 3), dtype=np.uint8)
image[:, :] = [200, 200, 200]

positions = [
    (50, 50), (350, 50), (350, 250), (50, 250),
    (200, 100), (200, 200), (100, 150), (300, 150)
]
for i, pos in enumerate(positions):
    cv2.circle(image, pos, 15, (0, 128, 255), -1)
    cv2.putText(image, str(i), (pos[0] - 5, pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

h, w = image.shape[:2]

# 真实变换
true_H = np.array([
    [0.9, -0.1, 30],
    [0.15, 0.85, 20],
    [0.0002, 0.0001, 1]
], dtype=np.float32)

transformed = cv2.warpPerspective(image, true_H, (w, h))

# 变换特征点
src_pts = np.array(positions, dtype=np.float32)
ones = np.ones((len(src_pts), 1), dtype=np.float32)
src_h = np.hstack([src_pts, ones])
dst_h = (true_H @ src_h.T).T
dst_pts = dst_h[:, :2] / dst_h[:, 2:3]

# 加噪声
np.random.seed(42)
noise = np.random.normal(0, 2, dst_pts.shape).astype(np.float32)
dst_noisy = dst_pts + noise

# 不同估计方法
H_ransac, mask_ransac = cv2.findHomography(src_pts, dst_noisy, cv2.RANSAC, 5.0)
H_lmeds, mask_lmeds = cv2.findHomography(src_pts, dst_noisy, cv2.LMEDS)
H_4pts = cv2.getPerspectiveTransform(src_pts[:4], dst_noisy[:4])

result_ransac = cv2.warpPerspective(image, H_ransac, (w, h))
result_lmeds = cv2.warpPerspective(image, H_lmeds, (w, h))
result_4pts = cv2.warpPerspective(image, H_4pts, (w, h))


def matrix_error(H_est, H_true):
    H_est_n = H_est / H_est[2, 2]
    H_true_n = H_true / H_true[2, 2]
    return np.linalg.norm(H_est_n - H_true_n, 'fro')


fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('多点单应性估计对比', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('源点')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('目标点 (含噪声)')
axes[0, 1].axis('off')

err_r = matrix_error(H_ransac, true_H)
axes[0, 2].imshow(cv2.cvtColor(result_ransac, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f'RANSAC\n误差: {err_r:.4f}')
axes[0, 2].axis('off')

err_l = matrix_error(H_lmeds, true_H)
axes[0, 3].imshow(cv2.cvtColor(result_lmeds, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title(f'LMEDS\n误差: {err_l:.4f}')
axes[0, 3].axis('off')

err_4 = matrix_error(H_4pts, true_H)
axes[1, 0].imshow(cv2.cvtColor(result_4pts, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f'4点精确解\n误差: {err_4:.4f}')
axes[1, 0].axis('off')

# 点对应
for i in range(len(src_pts)):
    axes[1, 1].plot([src_pts[i, 0], dst_noisy[i, 0]],
                    [src_pts[i, 1], dst_noisy[i, 1]], 'b-', alpha=0.5)
    axes[1, 1].plot(src_pts[i, 0], src_pts[i, 1], 'go', markersize=8)
    axes[1, 1].plot(dst_noisy[i, 0], dst_noisy[i, 1], 'rx', markersize=8)
axes[1, 1].invert_yaxis()
axes[1, 1].legend(['对应', '源', '目标'], fontsize=8)
axes[1, 1].set_title('点对应关系')
axes[1, 1].grid(True, alpha=0.3)

# 方法说明
info = f"RANSAC内点: {mask_ransac.sum()}/{len(mask_ransac)}\n"
info += f"LMEDS内点: {mask_lmeds.sum()}/{len(mask_lmeds)}\n\n"
info += "RANSAC: 随机采样一致性\n  适合有外点的情况\n\n"
info += "LMEDS: 最小中位数二乘\n  适合<50%外点\n\n"
info += "4点: 精确解\n  对噪声敏感"
axes[1, 2].text(0.1, 0.5, info, fontsize=10, family='monospace',
                verticalalignment='center', transform=axes[1, 2].transAxes)
axes[1, 2].axis('off')
axes[1, 2].set_title('方法比较')

# 误差柱状图
methods = ['RANSAC', 'LMEDS', '4点']
errors = [err_r, err_l, err_4]
axes[1, 3].bar(methods, errors, color=['green', 'blue', 'red'])
axes[1, 3].set_ylabel('Frobenius范数误差')
axes[1, 3].set_title('估计误差')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_homography_estimation.png'), dpi=150, bbox_inches='tight')
plt.show()

print("单应性估计对比完成！")
