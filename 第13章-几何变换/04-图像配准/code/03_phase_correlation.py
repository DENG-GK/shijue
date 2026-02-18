"""
示例3：相位相关法配准
- FFT相位相关检测纯平移
- 手动实现 + cv2.phaseCorrelate
- 适用于小位移场景
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建参考图像
reference = np.zeros((256, 256), dtype=np.uint8)
cv2.rectangle(reference, (50, 50), (200, 200), 200, -1)
cv2.circle(reference, (128, 128), 40, 150, -1)
np.random.seed(42)
for i in range(20):
    x, y = np.random.randint(60, 190), np.random.randint(60, 190)
    cv2.circle(reference, (x, y), 5, 100, -1)

# 平移
true_shift = (30, 20)
M = np.float32([[1, 0, true_shift[0]], [0, 1, true_shift[1]]])
source = cv2.warpAffine(reference, M, (256, 256))


def phase_correlate_manual(ref, src):
    """手动实现相位相关"""
    ref_f = ref.astype(np.float64)
    src_f = src.astype(np.float64)

    rows, cols = ref_f.shape
    hann = cv2.createHanningWindow((cols, rows), cv2.CV_64F)
    ref_f = ref_f * hann
    src_f = src_f * hann

    ref_fft = np.fft.fft2(ref_f)
    src_fft = np.fft.fft2(src_f)

    cross = ref_fft * np.conj(src_fft)
    magnitude = np.abs(cross)
    magnitude[magnitude == 0] = 1
    normalized = cross / magnitude

    correlation = np.abs(np.fft.ifft2(normalized))
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)

    sy = peak[0] - rows if peak[0] > rows // 2 else peak[0]
    sx = peak[1] - cols if peak[1] > cols // 2 else peak[1]

    return (-sx, -sy), correlation


estimated_shift, correlation = phase_correlate_manual(reference, source)
print(f"真实位移: {true_shift}")
print(f"手动估计: {estimated_shift}")

# OpenCV相位相关
shift_cv, response = cv2.phaseCorrelate(
    reference.astype(np.float64), source.astype(np.float64))
print(f"OpenCV估计: ({shift_cv[0]:.2f}, {shift_cv[1]:.2f})")

# 校正
M_correct = np.float32([[1, 0, estimated_shift[0]], [0, 1, estimated_shift[1]]])
aligned = cv2.warpAffine(source, M_correct, (256, 256))

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('相位相关法配准', fontsize=14, fontweight='bold')

axes[0, 0].imshow(reference, cmap='gray')
axes[0, 0].set_title('参考图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(source, cmap='gray')
axes[0, 1].set_title(f'源图像 (平移{true_shift})')
axes[0, 1].axis('off')

axes[0, 2].imshow(np.fft.fftshift(correlation), cmap='hot')
axes[0, 2].set_title('相位相关图')
axes[0, 2].axis('off')

axes[0, 3].imshow(aligned, cmap='gray')
axes[0, 3].set_title(f'对齐结果 (估计{estimated_shift})')
axes[0, 3].axis('off')

# RGB叠加
overlay_before = np.stack([reference, source, np.zeros_like(reference)], axis=2)
axes[1, 0].imshow(overlay_before)
axes[1, 0].set_title('对齐前 (R=参考, G=源)')
axes[1, 0].axis('off')

overlay_after = np.stack([reference, aligned, np.zeros_like(reference)], axis=2)
axes[1, 1].imshow(overlay_after)
axes[1, 1].set_title('对齐后 (应为黄色)')
axes[1, 1].axis('off')

error_before = cv2.absdiff(reference, source)
axes[1, 2].imshow(error_before, cmap='hot')
axes[1, 2].set_title(f'对齐前误差\n总和: {np.sum(error_before)}')
axes[1, 2].axis('off')

error_after = cv2.absdiff(reference, aligned)
axes[1, 3].imshow(error_after, cmap='hot')
axes[1, 3].set_title(f'对齐后误差\n总和: {np.sum(error_after)}')
axes[1, 3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_phase_correlation.png'), dpi=150, bbox_inches='tight')
plt.show()
