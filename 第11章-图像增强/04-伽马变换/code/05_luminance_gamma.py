"""
示例5：仅对亮度通道进行伽马变换
- 在LAB空间仅对L通道应用伽马变换
- 比直接对RGB处理更好地保持色彩饱和度
- 对比RGB方法和LAB方法的差异
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def gamma_transform_color(image, gamma):
    """直接对BGR三通道伽马变换"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def gamma_transform_luminance(image, gamma):
    """仅对亮度通道进行伽马变换（LAB空间）"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    l_corrected = cv2.LUT(l, table)

    lab_corrected = cv2.merge([l_corrected, a, b])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


# 创建彩色测试图像
def create_test_image():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    for i in range(300):
        for j in range(400):
            img[i, j] = [30 + int(30 * np.sin(i / 40)),
                        50 + int(25 * np.cos(j / 50)),
                        40 + int(20 * np.sin((i + j) / 60))]
    cv2.circle(img, (100, 150), 50, (80, 40, 120), -1)
    cv2.rectangle(img, (200, 80), (350, 220), (60, 130, 80), -1)
    return img


image = create_test_image()
gamma = 0.5

result_rgb = gamma_transform_color(image, gamma)
result_lab = gamma_transform_luminance(image, gamma)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('RGB伽马 vs LAB亮度伽马', fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(result_rgb, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'RGB伽马 (γ={gamma})\n可能改变色彩', fontsize=11)
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(result_lab, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'LAB亮度伽马 (γ={gamma})\n更好保持色彩', fontsize=11)
axes[2].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_luminance_gamma.png'), dpi=150, bbox_inches='tight')
plt.show()

print("LAB亮度通道伽马变换：")
print("- 只修改L通道，保持a、b通道不变")
print("- 比直接RGB方法更好地保持色彩饱和度")
print("- 推荐用于彩色图像的伽马校正")
