"""
示例7：配准误差分析
- 点误差、矩阵误差、图像误差
- MSE/PSNR 指标
- 误差分布直方图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建带网格标记的参考图像
np.random.seed(42)
reference = np.ones((300, 400, 3), dtype=np.uint8) * 200

markers = []
for i in range(4):
    for j in range(5):
        x = 50 + j * 75
        y = 50 + i * 65
        markers.append((x, y))
        cv2.circle(reference, (x, y), 10, (0, 0, 255), -1)
        cv2.circle(reference, (x, y), 4, (255, 255, 255), -1)

# 添加纹理
for _ in range(30):
    x, y = np.random.randint(30, 370), np.random.randint(30, 270)
    cv2.circle(reference, (x, y), 3, (150, 150, 150), -1)

h, w = reference.shape[:2]

# 变换后的源图像
angle, scale, tx, ty = 5, 0.95, 10, 8
M_true = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
M_true[0, 2] += tx
M_true[1, 2] += ty
source = cv2.warpAffine(reference, M_true, (w, h), borderValue=(180, 180, 180))

# ORB配准
gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
gray_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=500)
kp_ref, desc_ref = orb.detectAndCompute(gray_ref, None)
kp_src, desc_src = orb.detectAndCompute(gray_src, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc_src, desc_ref)
matches = sorted(matches, key=lambda x: x.distance)

src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

M_estimated, mask = cv2.estimateAffine2D(src_pts, ref_pts, method=cv2.RANSAC)
aligned = cv2.warpAffine(source, M_estimated, (w, h))

# 1. 矩阵误差
matrix_error = np.linalg.norm(M_true - M_estimated, 'fro')

# 2. 逐点误差
markers_arr = np.array(markers, dtype=np.float32)
ones = np.ones((len(markers), 1), dtype=np.float32)
markers_h = np.hstack([markers_arr, ones])

true_transformed = (M_true @ markers_h.T).T
est_transformed = (M_estimated @ markers_h.T).T
point_errors = np.sqrt(np.sum((true_transformed - est_transformed) ** 2, axis=1))

# 3. 图像误差
error_img = cv2.absdiff(reference, aligned)
mse = np.mean(error_img.astype(float) ** 2)
psnr = cv2.PSNR(reference, aligned)

print(f"矩阵Frobenius范数误差: {matrix_error:.6f}")
print(f"点误差 - 均值: {np.mean(point_errors):.3f}px, 最大: {np.max(point_errors):.3f}px")
print(f"MSE: {mse:.2f}, PSNR: {psnr:.2f} dB")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('配准误差分析', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('参考图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('源图像')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('对齐结果')
axes[0, 2].axis('off')

error_enh = cv2.convertScaleAbs(error_img, alpha=5)
axes[0, 3].imshow(cv2.cvtColor(error_enh, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title(f'误差 (5x)\nMSE: {mse:.2f}')
axes[0, 3].axis('off')

# 逐点误差可视化
max_err = max(point_errors) if max(point_errors) > 0 else 1
for marker, err in zip(markers, point_errors):
    color = plt.cm.hot(err / max_err)[:3]
    axes[1, 0].scatter(marker[0], marker[1], c=[color], s=100, edgecolors='black', linewidths=0.5)
axes[1, 0].set_xlim(0, w)
axes[1, 0].set_ylim(h, 0)
axes[1, 0].set_title('逐点误差分布')
axes[1, 0].set_aspect('equal')

# 误差直方图
axes[1, 1].hist(point_errors, bins=10, color='steelblue', edgecolor='black', alpha=0.8)
axes[1, 1].axvline(np.mean(point_errors), color='red', linestyle='--', label=f'均值: {np.mean(point_errors):.2f}px')
axes[1, 1].set_xlabel('误差 (像素)')
axes[1, 1].set_ylabel('计数')
axes[1, 1].set_title('误差分布直方图')
axes[1, 1].legend(fontsize=8)

# 矩阵对比
info = f"真实变换:\n[{M_true[0, 0]:.4f}  {M_true[0, 1]:.4f}  {M_true[0, 2]:.2f}]\n"
info += f"[{M_true[1, 0]:.4f}  {M_true[1, 1]:.4f}  {M_true[1, 2]:.2f}]\n\n"
info += f"估计变换:\n[{M_estimated[0, 0]:.4f}  {M_estimated[0, 1]:.4f}  {M_estimated[0, 2]:.2f}]\n"
info += f"[{M_estimated[1, 0]:.4f}  {M_estimated[1, 1]:.4f}  {M_estimated[1, 2]:.2f}]\n\n"
info += f"Frobenius范数: {matrix_error:.6f}"
axes[1, 2].text(0.05, 0.5, info, fontsize=9, family='monospace',
                verticalalignment='center', transform=axes[1, 2].transAxes)
axes[1, 2].axis('off')
axes[1, 2].set_title('矩阵误差')

# 汇总
summary = f"配准质量汇总\n\n"
summary += f"矩阵误差:\n  Frobenius: {matrix_error:.6f}\n\n"
summary += f"逐点误差:\n  均值: {np.mean(point_errors):.3f} px\n"
summary += f"  最大: {np.max(point_errors):.3f} px\n"
summary += f"  标准差: {np.std(point_errors):.3f} px\n\n"
summary += f"图像误差:\n  MSE:  {mse:.2f}\n  PSNR: {psnr:.2f} dB\n\n"
summary += f"匹配数: {len(matches)}\n"
summary += f"内点数: {mask.sum()}"
axes[1, 3].text(0.05, 0.5, summary, fontsize=9, family='monospace',
                verticalalignment='center', transform=axes[1, 3].transAxes)
axes[1, 3].axis('off')
axes[1, 3].set_title('质量汇总')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_error_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()

print("配准误差分析完成！")
