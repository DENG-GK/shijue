"""
示例8：彩色图像的CLAHE处理
- 支持LAB、YCrCb、HSV三种颜色空间
- LAB方法最推荐（L通道与人眼感知最接近）
- 对比三种方法的效果差异
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def apply_clahe_color(image, clip_limit=2.0, tile_size=(8, 8), method='lab'):
    """
    对彩色图像应用CLAHE

    Parameters:
    -----------
    method : str
        'lab' - 在LAB空间的L通道应用 (推荐)
        'ycrcb' - 在YCrCb空间的Y通道应用
        'hsv' - 在HSV空间的V通道应用
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    if method == 'lab':
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    elif method == 'ycrcb':
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = clahe.apply(y)
        ycrcb = cv2.merge([y, cr, cb])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    elif method == 'hsv':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    else:
        raise ValueError(f"未知方法: {method}")

    return result


# 创建低对比度彩色图像
def create_low_contrast_color():
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    for i in range(300):
        for j in range(400):
            base = 80 + int(40 * np.sin(i / 50))
            img[i, j] = [base, base + 20, base + 10]

    cv2.circle(img, (100, 150), 50, (120, 80, 100), -1)
    cv2.rectangle(img, (180, 80), (300, 220), (100, 130, 90), -1)

    return img


color_img = create_low_contrast_color()

# 应用不同方法
result_lab = apply_clahe_color(color_img, method='lab')
result_ycrcb = apply_clahe_color(color_img, method='ycrcb')
result_hsv = apply_clahe_color(color_img, method='hsv')

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('彩色图像CLAHE - 三种颜色空间对比', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(result_lab, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('CLAHE (LAB) - 推荐', fontsize=12)
axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(result_ycrcb, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('CLAHE (YCrCb)', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(result_hsv, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('CLAHE (HSV)', fontsize=12)
axes[1, 1].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_clahe_color.png'), dpi=150, bbox_inches='tight')
plt.show()

print("彩色图像CLAHE方法对比：")
print("- LAB: 最推荐，L通道与人眼感知最接近")
print("- YCrCb: 效果良好，计算简单")
print("- HSV: 可能在极端情况下产生颜色偏移")
