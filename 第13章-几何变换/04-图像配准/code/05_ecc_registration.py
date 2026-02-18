"""
示例5：增强相关系数法（ECC）配准
- findTransformECC 迭代优化
- Translation/Euclidean/Affine 运动模型对比
- 适用于精细配准场景
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建参考图像
np.random.seed(42)
reference = np.zeros((256, 256, 3), dtype=np.uint8)
reference[:, :] = [200, 200, 200]
cv2.rectangle(reference, (40, 40), (216, 216), (0, 128, 255), -1)
cv2.circle(reference, (128, 100), 30, (255, 0, 0), -1)
cv2.ellipse(reference, (128, 180), (50, 25), 0, 0, 360, (0, 255, 0), -1)
for i in range(30):
    x, y = np.random.randint(50, 200), np.random.randint(50, 200)
    cv2.circle(reference, (x, y), 3, (100, 100, 100), -1)

h, w = reference.shape[:2]

# 变换后的源图像
angle, scale, tx, ty = 8, 0.92, 12, 8
M_true = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
M_true[0, 2] += tx
M_true[1, 2] += ty
source = cv2.warpAffine(reference, M_true, (w, h), borderValue=(128, 128, 128))

# 灰度转换
gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)

# ECC参数
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)

# 不同运动模型
models = {
    'Translation': cv2.MOTION_TRANSLATION,
    'Euclidean': cv2.MOTION_EUCLIDEAN,
    'Affine': cv2.MOTION_AFFINE,
}

results = {}
for name, motion_type in models.items():
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp_matrix = cv2.findTransformECC(
            gray_ref, gray_src, warp_matrix, motion_type, criteria)
        aligned = cv2.warpAffine(source, warp_matrix, (w, h),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        results[name] = {'cc': cc, 'matrix': warp_matrix, 'aligned': aligned}
        print(f"{name}: CC = {cc:.6f}")
    except cv2.error as e:
        print(f"{name}: 失败 - {e}")
        results[name] = None

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('ECC增强相关系数法配准', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('参考图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('源图像')
axes[0, 1].axis('off')

for i, (name, result) in enumerate(results.items()):
    if result is not None:
        axes[0, 2 + i].imshow(cv2.cvtColor(result['aligned'], cv2.COLOR_BGR2RGB))
        axes[0, 2 + i].set_title(f'{name}\nCC: {result["cc"]:.4f}')
    axes[0, 2 + i].axis('off')

# 叠加对比
overlay_before = cv2.addWeighted(reference, 0.5, source, 0.5, 0)
axes[1, 0].imshow(cv2.cvtColor(overlay_before, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('对齐前叠加')
axes[1, 0].axis('off')

if results.get('Affine') is not None:
    overlay_after = cv2.addWeighted(reference, 0.5, results['Affine']['aligned'], 0.5, 0)
    axes[1, 1].imshow(cv2.cvtColor(overlay_after, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('对齐后 (Affine)')
axes[1, 1].axis('off')

# 误差图
if results.get('Affine') is not None:
    error = cv2.absdiff(reference, results['Affine']['aligned'])
    error_enh = cv2.convertScaleAbs(error, alpha=5)
    axes[1, 2].imshow(cv2.cvtColor(error_enh, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'误差 (5x)\n均值: {np.mean(error):.2f}')
axes[1, 2].axis('off')

# 矩阵对比
info = f"真实变换:\n[{M_true[0, 0]:.4f}  {M_true[0, 1]:.4f}  {M_true[0, 2]:.2f}]\n"
info += f"[{M_true[1, 0]:.4f}  {M_true[1, 1]:.4f}  {M_true[1, 2]:.2f}]\n\n"
if results.get('Affine') is not None:
    M_est = results['Affine']['matrix']
    info += f"ECC估计 (Affine):\n[{M_est[0, 0]:.4f}  {M_est[0, 1]:.4f}  {M_est[0, 2]:.2f}]\n"
    info += f"[{M_est[1, 0]:.4f}  {M_est[1, 1]:.4f}  {M_est[1, 2]:.2f}]\n\n"
    info += f"矩阵误差: {np.linalg.norm(M_true - M_est):.6f}"
axes[1, 3].text(0.1, 0.5, info, fontsize=9, family='monospace',
                verticalalignment='center', transform=axes[1, 3].transAxes)
axes[1, 3].axis('off')
axes[1, 3].set_title('矩阵对比')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_ecc_registration.png'), dpi=150, bbox_inches='tight')
plt.show()
