"""
示例3：手动实现金字塔操作
- 5×5高斯核（OpenCV标准核）
- 手动pyrDown：高斯滤波 + 隔行隔列采样
- 手动pyrUp：插入零值 + 高斯滤波×4
- 与OpenCV实现的对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# OpenCV使用的5×5高斯核
kernel = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], dtype=np.float64) / 256.0

print("高斯核:")
print(kernel)
print(f"核的和: {kernel.sum():.4f}")

# 创建测试图像
image = np.zeros((256, 256), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (200, 200), 200, -1)
cv2.circle(image, (128, 128), 40, 100, -1)


def manual_pyr_down(img):
    """手动下采样"""
    filtered = cv2.filter2D(img.astype(np.float64), -1, kernel)
    return filtered[::2, ::2].astype(np.uint8)


def manual_pyr_up(img):
    """手动上采样"""
    h, w = img.shape[:2]
    upsampled = np.zeros((h * 2, w * 2), dtype=np.float64)
    upsampled[::2, ::2] = img
    filtered = cv2.filter2D(upsampled, -1, kernel) * 4
    return np.clip(filtered, 0, 255).astype(np.uint8)


# 对比OpenCV和手动实现
opencv_down = cv2.pyrDown(image)
manual_down = manual_pyr_down(image)
opencv_up = cv2.pyrUp(opencv_down)
manual_up = manual_pyr_up(opencv_down)

diff_down = cv2.absdiff(opencv_down, manual_down)
diff_up = cv2.absdiff(opencv_up, manual_up)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('手动实现 vs OpenCV 金字塔操作', fontsize=14, fontweight='bold')

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title(f'原图 {image.shape}')
axes[0, 0].axis('off')

axes[0, 1].imshow(opencv_down, cmap='gray')
axes[0, 1].set_title(f'OpenCV pyrDown\n{opencv_down.shape}')
axes[0, 1].axis('off')

axes[0, 2].imshow(manual_down, cmap='gray')
axes[0, 2].set_title(f'手动 pyrDown\n{manual_down.shape}')
axes[0, 2].axis('off')

axes[0, 3].imshow(diff_down, cmap='hot')
axes[0, 3].set_title(f'差异 (Down)\nMax: {diff_down.max()}')
axes[0, 3].axis('off')

axes[1, 0].imshow(opencv_down, cmap='gray')
axes[1, 0].set_title('下采样结果')
axes[1, 0].axis('off')

axes[1, 1].imshow(opencv_up, cmap='gray')
axes[1, 1].set_title(f'OpenCV pyrUp\n{opencv_up.shape}')
axes[1, 1].axis('off')

axes[1, 2].imshow(manual_up, cmap='gray')
axes[1, 2].set_title(f'手动 pyrUp\n{manual_up.shape}')
axes[1, 2].axis('off')

axes[1, 3].imshow(diff_up, cmap='hot')
axes[1, 3].set_title(f'差异 (Up)\nMax: {diff_up.max()}')
axes[1, 3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_manual_pyramid.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\npyrDown差异 - Max: {diff_down.max()}, Mean: {diff_down.mean():.2f}")
print(f"pyrUp差异 - Max: {diff_up.max()}, Mean: {diff_up.mean():.2f}")
