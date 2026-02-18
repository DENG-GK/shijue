"""
示例10：夜景/低光照图像增强
- 在YCrCb空间仅处理亮度通道
- 混合原始与均衡化结果避免过度增强
- 创建模拟夜景图像进行演示
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def enhance_night_image(image):
    """夜景图像增强"""
    # 转换到YCrCb
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
    else:
        y = image.copy()

    # 对Y通道进行处理
    # 1. 轻微降噪
    y_denoised = cv2.GaussianBlur(y, (3, 3), 0)

    # 2. 直方图均衡化
    y_equalized = cv2.equalizeHist(y_denoised)

    # 3. 混合原始和均衡化结果（避免过度增强）
    alpha = 0.7
    y_blended = cv2.addWeighted(y_equalized, alpha, y, 1 - alpha, 0)

    if len(image.shape) == 3:
        enhanced = cv2.merge([y_blended, cr, cb])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)
    else:
        enhanced = y_blended

    return enhanced


def create_night_image():
    """创建模拟夜景图像"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # 深色背景
    img[:] = [20, 25, 30]

    # 添加一些"灯光"
    lights = [(80, 100), (200, 80), (320, 120), (150, 200), (280, 180)]
    for x, y in lights:
        for r in range(50, 0, -5):
            intensity = 30 + (50 - r) * 2
            cv2.circle(img, (x, y), r, (intensity, intensity + 10, intensity + 20), -1)

    # 添加一些建筑轮廓
    cv2.rectangle(img, (50, 150), (120, 280), (40, 35, 30), -1)
    cv2.rectangle(img, (180, 120), (240, 280), (35, 40, 35), -1)
    cv2.rectangle(img, (280, 100), (370, 280), (30, 35, 40), -1)

    return img


night_img = create_night_image()
enhanced = enhance_night_image(night_img)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('夜景/低光照图像增强', fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(night_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('原始夜景图像', fontsize=12)
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
axes[1].set_title('增强后的夜景图像', fontsize=12)
axes[1].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_night_enhancement.png'), dpi=150, bbox_inches='tight')
plt.show()

print("夜景图像增强技巧：")
print("1. 在YCrCb/LAB空间处理，保持颜色")
print("2. 只增强亮度通道(Y/L)")
print("3. 混合原始和增强结果，避免过度处理")
print("4. 可配合降噪使用")
