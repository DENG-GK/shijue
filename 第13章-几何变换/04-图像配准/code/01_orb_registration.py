"""
示例1：ORB特征配准
- ORB特征点检测与匹配
- BFMatcher 暴力匹配
- estimateAffine2D (RANSAC)
- 配准前后叠加对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建参考图像
reference = np.ones((300, 400, 3), dtype=np.uint8) * 200
cv2.rectangle(reference, (50, 50), (350, 250), (0, 128, 255), -1)
cv2.circle(reference, (100, 100), 30, (255, 0, 0), -1)
cv2.circle(reference, (300, 100), 30, (0, 255, 0), -1)
cv2.circle(reference, (200, 200), 30, (0, 0, 255), -1)
cv2.putText(reference, 'REF', (160, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# 添加纹理
np.random.seed(42)
for i in range(20):
    x, y = np.random.randint(60, 340), np.random.randint(60, 240)
    cv2.circle(reference, (x, y), 5, (150, 150, 150), -1)

h, w = reference.shape[:2]

# 变换后的源图像
angle, scale, tx, ty = 15, 0.9, 30, 20
M_true = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
M_true[0, 2] += tx
M_true[1, 2] += ty
source = cv2.warpAffine(reference, M_true, (w, h), borderValue=(128, 128, 128))

# ORB检测与匹配
gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
gray_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=500)
kp_ref, desc_ref = orb.detectAndCompute(gray_ref, None)
kp_src, desc_src = orb.detectAndCompute(gray_src, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc_src, desc_ref)
matches = sorted(matches, key=lambda x: x.distance)

print(f"参考关键点: {len(kp_ref)}, 源关键点: {len(kp_src)}, 匹配: {len(matches)}")

# 估计变换
src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M_est, mask = cv2.estimateAffine2D(src_pts, ref_pts, method=cv2.RANSAC)
aligned = cv2.warpAffine(source, M_est, (w, h))

error = cv2.absdiff(reference, aligned)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('ORB特征配准', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('参考图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('源图像 (变换后)')
axes[0, 1].axis('off')

match_img = cv2.drawMatches(source, kp_src, reference, kp_ref,
                             matches[:20], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
axes[0, 2].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f'特征匹配 (前20/{len(matches)})')
axes[0, 2].axis('off')

axes[0, 3].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title('对齐结果')
axes[0, 3].axis('off')

overlay_before = cv2.addWeighted(reference, 0.5, source, 0.5, 0)
axes[1, 0].imshow(cv2.cvtColor(overlay_before, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('对齐前叠加')
axes[1, 0].axis('off')

overlay_after = cv2.addWeighted(reference, 0.5, aligned, 0.5, 0)
axes[1, 1].imshow(cv2.cvtColor(overlay_after, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('对齐后叠加')
axes[1, 1].axis('off')

error_enh = cv2.convertScaleAbs(error, alpha=3)
axes[1, 2].imshow(cv2.cvtColor(error_enh, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f'误差 (3x)\n均值: {np.mean(error):.2f}')
axes[1, 2].axis('off')

# 矩阵对比
info = f"真实变换:\n[{M_true[0, 0]:.3f} {M_true[0, 1]:.3f} {M_true[0, 2]:.1f}]\n"
info += f"[{M_true[1, 0]:.3f} {M_true[1, 1]:.3f} {M_true[1, 2]:.1f}]\n\n"
info += f"估计变换:\n[{M_est[0, 0]:.3f} {M_est[0, 1]:.3f} {M_est[0, 2]:.1f}]\n"
info += f"[{M_est[1, 0]:.3f} {M_est[1, 1]:.3f} {M_est[1, 2]:.1f}]\n\n"
info += f"矩阵误差: {np.linalg.norm(M_true - M_est):.4f}"
axes[1, 3].text(0.1, 0.5, info, fontsize=9, family='monospace',
                verticalalignment='center', transform=axes[1, 3].transAxes)
axes[1, 3].axis('off')
axes[1, 3].set_title('变换矩阵对比')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_orb_registration.png'), dpi=150, bbox_inches='tight')
plt.show()
