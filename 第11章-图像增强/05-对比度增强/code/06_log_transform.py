"""
示例6：对数变换
- 公式：s = c * log(1 + r)
- 扩展暗部细节，压缩亮部范围
- 对比对数变换和反对数变换
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def log_transform(image, c=None):
    """对数变换 s = c * log(1 + r)"""
    img_float = image.astype(np.float64)
    log_img = np.log1p(img_float)
    if c is None:
        c = 255 / np.log1p(255)
    result = c * log_img
    return np.clip(result, 0, 255).astype(np.uint8)


def inverse_log_transform(image):
    """反对数变换"""
    img_float = image.astype(np.float64) / 255.0
    exp_img = np.expm1(img_float * np.log(256))
    c = 255 / np.expm1(np.log(256))
    result = c * exp_img
    return np.clip(result, 0, 255).astype(np.uint8)


# 创建高动态范围测试图像
image = np.zeros((300, 400), dtype=np.uint8)
image[:, :200] = np.random.randint(0, 50, (300, 200), dtype=np.uint8)
image[:, 200:] = np.random.randint(200, 256, (300, 200), dtype=np.uint8)

log_result = log_transform(image)
inv_log_result = inverse_log_transform(image)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('对数变换与反对数变换', fontsize=14, fontweight='bold')

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('原图')
axes[0, 0].axis('off')
axes[0, 1].imshow(log_result, cmap='gray')
axes[0, 1].set_title('对数变换（增强暗部）')
axes[0, 1].axis('off')
axes[0, 2].imshow(inv_log_result, cmap='gray')
axes[0, 2].set_title('反对数变换（增强亮部）')
axes[0, 2].axis('off')

# 变换曲线
x = np.arange(256)
c_log = 255 / np.log1p(255)
y_log = c_log * np.log1p(x)

axes[1, 0].plot(x, x, 'k--', label='线性', alpha=0.5)
axes[1, 0].plot(x, y_log, 'b-', label='对数', linewidth=2)
axes[1, 0].set_title('变换曲线')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(image.flatten(), bins=256, range=[0, 256], alpha=0.7)
axes[1, 1].set_title('原始直方图')
axes[1, 2].hist(log_result.flatten(), bins=256, range=[0, 256], alpha=0.7)
axes[1, 2].set_title('对数变换直方图')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_log_transform.png'), dpi=150, bbox_inches='tight')
plt.show()

print("对数变换：扩展暗部，压缩亮部（适合高动态范围图像）")
print("反对数变换：扩展亮部，压缩暗部")
