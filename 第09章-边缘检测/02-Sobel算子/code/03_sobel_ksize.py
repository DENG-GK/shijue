"""
示例3：对比不同ksize参数对Sobel边缘检测的影响
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建带噪声的测试图像 =====================

def create_noisy_image():
    """创建一个带噪声的图像"""
    # 基础图像
    img = np.zeros((250, 350), dtype=np.uint8)
    img[:] = 80

    # 添加形状
    cv2.rectangle(img, (50, 50), (150, 150), 180, -1)
    cv2.circle(img, (250, 100), 50, 180, -1)

    # 斜线
    cv2.line(img, (50, 180), (150, 230), 180, 4)

    # 添加高斯噪声
    noise = np.random.normal(0, 15, img.shape).astype(np.float64)
    noisy = img.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return img, noisy

clean_img, noisy_img = create_noisy_image()

print("测试图像：")
print(f"  干净图像")
print(f"  带噪声图像（高斯噪声，标准差=15）")

# ===================== 不同ksize的Sobel =====================

ksize_list = [1, 3, 5, 7]
results = {}

for ksize in ksize_list:
    # 对噪声图像应用Sobel
    sobel_x = cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=ksize)

    # 计算梯度幅值
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    results[ksize] = magnitude

# 也测试ksize=-1（Scharr）
sobel_x_scharr = cv2.Sobel(noisy_img, cv2.CV_64F, 1, 0, ksize=-1)
sobel_y_scharr = cv2.Sobel(noisy_img, cv2.CV_64F, 0, 1, ksize=-1)
magnitude_scharr = np.sqrt(sobel_x_scharr**2 + sobel_y_scharr**2)
magnitude_scharr = np.clip(magnitude_scharr, 0, 255).astype(np.uint8)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 第一行：原图和ksize 1, 3
axes[0, 0].imshow(clean_img, cmap='gray')
axes[0, 0].set_title('原始图像（无噪声）', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(noisy_img, cmap='gray')
axes[0, 1].set_title('带噪声图像', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(results[1], cmap='gray')
axes[0, 2].set_title('ksize=1\n（最简单，噪声明显）', fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(results[3], cmap='gray')
axes[0, 3].set_title('ksize=3\n（标准Sobel）', fontsize=11)
axes[0, 3].axis('off')

# 第二行：ksize 5, 7, Scharr和说明
axes[1, 0].imshow(results[5], cmap='gray')
axes[1, 0].set_title('ksize=5\n（更多平滑）', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(results[7], cmap='gray')
axes[1, 1].set_title('ksize=7\n（最多平滑，边缘变粗）', fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(magnitude_scharr, cmap='gray')
axes[1, 2].set_title('ksize=-1 (Scharr)\n（更精确）', fontsize=11)
axes[1, 2].axis('off')

# 说明文字
axes[1, 3].axis('off')
info_text = """
ksize 选择指南：

ksize=1:
  最简单的差分
  对噪声非常敏感

ksize=3:
  标准Sobel核
  适合大多数场景
  推荐作为默认选择

ksize=5:
  更强的平滑效果
  抗噪声能力更好

ksize=7:
  最强的平滑
  边缘会变粗

ksize=-1 (Scharr):
  精度比3x3 Sobel高
  对角方向效果更好
"""
axes[1, 3].text(0.1, 0.5, info_text, fontsize=9,
                verticalalignment='center', fontfamily='SimHei',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('不同ksize对Sobel边缘检测的影响', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_ksize.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图像已保存为 'sobel_ksize.png'")
