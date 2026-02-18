"""
示例9：Retinex算法
- SSR：单尺度Retinex
- MSR：多尺度Retinex
- MSRCR：带颜色恢复的多尺度Retinex
- 对比不同sigma和方法的效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def single_scale_retinex(image, sigma=30):
    """单尺度Retinex (SSR)"""
    img_float = image.astype(np.float64) + 1
    log_img = np.log(img_float)

    # 高斯模糊估计光照分量
    blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
    log_blur = np.log(blur)

    # Retinex: log(反射) = log(图像) - log(光照)
    retinex = log_img - log_blur

    # 归一化到0-255
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255
    return retinex.astype(np.uint8)


def multi_scale_retinex(image, sigmas=[15, 80, 250]):
    """多尺度Retinex (MSR)"""
    img_float = image.astype(np.float64) + 1
    log_img = np.log(img_float)

    retinex = np.zeros_like(img_float)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
        retinex += log_img - np.log(blur)
    retinex /= len(sigmas)

    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255
    return retinex.astype(np.uint8)


def msrcr(image, sigmas=[15, 80, 250], alpha=125, beta=46, G=192, b=-30):
    """带颜色恢复的多尺度Retinex (MSRCR)"""
    img_float = image.astype(np.float64) + 1
    result = np.zeros_like(img_float)

    for i in range(3):
        channel = img_float[:, :, i]
        log_channel = np.log(channel)

        retinex = np.zeros_like(channel)
        for sigma in sigmas:
            blur = cv2.GaussianBlur(channel, (0, 0), sigma)
            retinex += log_channel - np.log(blur)
        retinex /= len(sigmas)

        # 颜色恢复因子
        sum_channels = np.sum(img_float, axis=2) + 1e-5
        color_restoration = beta * np.log(alpha * channel / sum_channels)

        result[:, :, i] = G * (retinex * color_restoration + b)

    result = (result - result.min()) / (result.max() - result.min()) * 255
    return result.astype(np.uint8)


# 创建测试图像（模拟光照不均）
image = np.zeros((300, 400, 3), dtype=np.uint8)
image[:, :200] = [30, 40, 50]
image[:, 200:] = [180, 190, 200]
cv2.circle(image, (100, 150), 50, (60, 80, 100), -1)
cv2.circle(image, (300, 150), 50, (150, 170, 190), -1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用不同方法
ssr_result = single_scale_retinex(gray)
msr_result = multi_scale_retinex(gray)
msrcr_result = msrcr(image)

# 不同sigma的SSR对比
ssr_15 = single_scale_retinex(gray, sigma=15)
ssr_80 = single_scale_retinex(gray, sigma=80)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Retinex 算法对比', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原图（彩色）')
axes[0, 0].axis('off')

axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('原图（灰度）')
axes[0, 1].axis('off')

axes[0, 2].imshow(ssr_result, cmap='gray')
axes[0, 2].set_title('SSR (σ=30)')
axes[0, 2].axis('off')

axes[0, 3].imshow(msr_result, cmap='gray')
axes[0, 3].set_title('MSR (多尺度)')
axes[0, 3].axis('off')

axes[1, 0].imshow(cv2.cvtColor(msrcr_result, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('MSRCR（彩色恢复）')
axes[1, 0].axis('off')

axes[1, 1].imshow(ssr_15, cmap='gray')
axes[1, 1].set_title('SSR (σ=15)')
axes[1, 1].axis('off')

axes[1, 2].imshow(ssr_80, cmap='gray')
axes[1, 2].set_title('SSR (σ=80)')
axes[1, 2].axis('off')

# 直方图对比
axes[1, 3].hist(gray.flatten(), bins=256, range=[0, 256], alpha=0.5, label='原图')
axes[1, 3].hist(msr_result.flatten(), bins=256, range=[0, 256], alpha=0.5, label='MSR')
axes[1, 3].set_title('直方图对比')
axes[1, 3].set_xlabel('灰度值')
axes[1, 3].legend(fontsize=8)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_retinex.png'), dpi=150, bbox_inches='tight')
plt.show()

print("SSR：单尺度，适合局部对比度增强")
print("MSR：多尺度，平衡增强效果")
print("MSRCR：包含颜色恢复，适合彩色图像")
