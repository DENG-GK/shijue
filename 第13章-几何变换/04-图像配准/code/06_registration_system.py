"""
示例6：综合配准系统
- ImageRegistrationSystem 类
- 支持特征匹配、ECC、相位相关三种方法
- MSE/PSNR/SSIM 质量评估
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ImageRegistrationSystem:
    """综合图像配准系统"""

    def __init__(self):
        self.reference = None
        self.source = None
        self.transform_matrix = None
        self.aligned = None

    def set_images(self, reference, source):
        """设置参考图像和源图像"""
        self.reference = reference
        self.source = source
        self.transform_matrix = None
        self.aligned = None

    def register_features(self, method='orb', transform='affine'):
        """基于特征的配准"""
        gray_ref = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        gray_src = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)

        if method == 'orb':
            detector = cv2.ORB_create(nfeatures=1000)
            norm = cv2.NORM_HAMMING
        else:
            try:
                detector = cv2.SIFT_create()
                norm = cv2.NORM_L2
            except AttributeError:
                detector = cv2.ORB_create(nfeatures=1000)
                norm = cv2.NORM_HAMMING

        kp_ref, desc_ref = detector.detectAndCompute(gray_ref, None)
        kp_src, desc_src = detector.detectAndCompute(gray_src, None)

        bf = cv2.BFMatcher(norm, crossCheck=True)
        matches = bf.match(desc_src, desc_ref)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        h, w = self.reference.shape[:2]
        if transform == 'affine':
            self.transform_matrix, mask = cv2.estimateAffine2D(
                src_pts, ref_pts, method=cv2.RANSAC)
            self.aligned = cv2.warpAffine(self.source, self.transform_matrix, (w, h))
        else:
            self.transform_matrix, mask = cv2.findHomography(
                src_pts, ref_pts, cv2.RANSAC)
            self.aligned = cv2.warpPerspective(self.source, self.transform_matrix, (w, h))

        return len(matches), mask.sum() if mask is not None else 0

    def register_ecc(self, motion_type=cv2.MOTION_AFFINE):
        """基于ECC的配准"""
        gray_ref = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_src = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        cc, self.transform_matrix = cv2.findTransformECC(
            gray_ref, gray_src, warp_matrix, motion_type, criteria)

        h, w = self.reference.shape[:2]
        self.aligned = cv2.warpAffine(self.source, self.transform_matrix, (w, h),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return cc

    def register_phase(self):
        """基于相位相关的配准（仅平移）"""
        gray_ref = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_src = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY).astype(np.float64)

        shift, response = cv2.phaseCorrelate(gray_ref, gray_src)
        self.transform_matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])

        h, w = self.reference.shape[:2]
        self.aligned = cv2.warpAffine(self.source, self.transform_matrix, (w, h))
        return shift, response

    def calculate_metrics(self):
        """计算配准质量指标"""
        if self.aligned is None:
            return None

        # MSE
        mse = np.mean((self.reference.astype(float) - self.aligned.astype(float)) ** 2)

        # PSNR
        psnr = cv2.PSNR(self.reference, self.aligned)

        # 简化SSIM
        gray_ref = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY).astype(float)
        gray_aligned = cv2.cvtColor(self.aligned, cv2.COLOR_BGR2GRAY).astype(float)

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mu_ref = cv2.GaussianBlur(gray_ref, (11, 11), 1.5)
        mu_aligned = cv2.GaussianBlur(gray_aligned, (11, 11), 1.5)

        sigma_ref_sq = cv2.GaussianBlur(gray_ref ** 2, (11, 11), 1.5) - mu_ref ** 2
        sigma_aligned_sq = cv2.GaussianBlur(gray_aligned ** 2, (11, 11), 1.5) - mu_aligned ** 2
        sigma_cross = cv2.GaussianBlur(gray_ref * gray_aligned, (11, 11), 1.5) - mu_ref * mu_aligned

        ssim_map = ((2 * mu_ref * mu_aligned + c1) * (2 * sigma_cross + c2)) / \
                   ((mu_ref ** 2 + mu_aligned ** 2 + c1) * (sigma_ref_sq + sigma_aligned_sq + c2))
        ssim = np.mean(ssim_map)

        return {'mse': mse, 'psnr': psnr, 'ssim': ssim}

    def visualize(self, title="配准结果"):
        """可视化配准结果"""
        if self.aligned is None:
            print("尚未执行配准")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        axes[0, 0].imshow(cv2.cvtColor(self.reference, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('参考图像')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cv2.cvtColor(self.source, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('源图像')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(cv2.cvtColor(self.aligned, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('对齐结果')
        axes[0, 2].axis('off')

        overlay_before = cv2.addWeighted(self.reference, 0.5, self.source, 0.5, 0)
        axes[1, 0].imshow(cv2.cvtColor(overlay_before, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('对齐前叠加')
        axes[1, 0].axis('off')

        overlay_after = cv2.addWeighted(self.reference, 0.5, self.aligned, 0.5, 0)
        axes[1, 1].imshow(cv2.cvtColor(overlay_after, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('对齐后叠加')
        axes[1, 1].axis('off')

        metrics = self.calculate_metrics()
        if metrics:
            info = f"质量评估指标:\n\n"
            info += f"MSE:  {metrics['mse']:.4f}\n"
            info += f"PSNR: {metrics['psnr']:.2f} dB\n"
            info += f"SSIM: {metrics['ssim']:.4f}"
            axes[1, 2].text(0.1, 0.5, info, fontsize=12, family='monospace',
                            verticalalignment='center', transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('质量指标')

        plt.tight_layout()
        return fig


# 演示
np.random.seed(42)
reference = np.ones((256, 256, 3), dtype=np.uint8) * 200
cv2.rectangle(reference, (40, 40), (216, 216), (0, 128, 255), -1)
cv2.circle(reference, (128, 100), 30, (255, 0, 0), -1)
for i in range(20):
    x, y = np.random.randint(50, 200), np.random.randint(50, 200)
    cv2.circle(reference, (x, y), 4, (100, 100, 100), -1)

h, w = reference.shape[:2]
M = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 0.95)
M[0, 2] += 15
M[1, 2] += 10
source = cv2.warpAffine(reference, M, (w, h), borderValue=(150, 150, 150))

system = ImageRegistrationSystem()

# 特征配准
system.set_images(reference, source)
n_matches, n_inliers = system.register_features(method='orb', transform='affine')
print(f"特征配准: {n_matches} 匹配, {n_inliers} 内点")
metrics_feat = system.calculate_metrics()
print(f"  MSE={metrics_feat['mse']:.2f}, PSNR={metrics_feat['psnr']:.2f}dB, SSIM={metrics_feat['ssim']:.4f}")
fig1 = system.visualize("特征配准 (ORB)")

# ECC配准
system.set_images(reference, source)
cc = system.register_ecc()
print(f"ECC配准: CC = {cc:.6f}")
metrics_ecc = system.calculate_metrics()
print(f"  MSE={metrics_ecc['mse']:.2f}, PSNR={metrics_ecc['psnr']:.2f}dB, SSIM={metrics_ecc['ssim']:.4f}")
fig2 = system.visualize("ECC配准")

# 相位相关
system.set_images(reference, source)
shift, response = system.register_phase()
print(f"相位相关: 位移=({shift[0]:.2f}, {shift[1]:.2f}), 响应={response:.4f}")
metrics_phase = system.calculate_metrics()
print(f"  MSE={metrics_phase['mse']:.2f}, PSNR={metrics_phase['psnr']:.2f}dB, SSIM={metrics_phase['ssim']:.4f}")

# 汇总对比
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle('三种配准方法对比', fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
axes[0].set_title('参考图像')
axes[0].axis('off')

labels = ['特征 (ORB)', 'ECC', '相位相关']
all_metrics = [metrics_feat, metrics_ecc, metrics_phase]

for i, (label, met) in enumerate(zip(labels, all_metrics)):
    info = f"{label}\n\nMSE:  {met['mse']:.2f}\nPSNR: {met['psnr']:.2f} dB\nSSIM: {met['ssim']:.4f}"
    axes[i + 1].text(0.1, 0.5, info, fontsize=11, family='monospace',
                     verticalalignment='center', transform=axes[i + 1].transAxes)
    axes[i + 1].axis('off')
    axes[i + 1].set_title(label)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
fig1.savefig(os.path.join(save_dir, '06_registration_system_feat.png'), dpi=150, bbox_inches='tight')
fig2.savefig(os.path.join(save_dir, '06_registration_system_ecc.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(save_dir, '06_registration_system.png'), dpi=150, bbox_inches='tight')
plt.show()

print("综合配准系统演示完成！")
